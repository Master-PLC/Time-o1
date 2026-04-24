import torch

import numpy as np

class fourier:
    '''

    num_freqs: number of frequencies assumed to be present in data
        type: int

    device: The device on which the computations are carried out.
        Example: cpu, cuda:0
        default = 'cpu'

    '''

    def __init__(self, num_freqs, device='cpu'):

        self.num_freqs = num_freqs
        self.device = device

    def fft(self, xt):
        '''
        Given temporal data xt, fft performs the initial guess of the 
        frequencies contained in the data using the FFT.

        Parameters
        ----------
        xt : TYPE: numpy.array
            Temporal data of dimensions [T, ...]

        Returns
        -------
        None.

        '''

        k = self.num_freqs
        self.freqs = []

        for i in range(k):

            N = len(xt)

            if len(self.freqs) == 0:
                residual = xt
            else:
                t = np.expand_dims(np.arange(N)+1, -1)
                freqs = np.array(self.freqs)
                Omega = np.concatenate(
                    [np.cos(t*2*np.pi*freqs), np.sin(t*2*np.pi*freqs)], -1)
                self.A = np.dot(np.linalg.pinv(Omega), xt)

                pred = np.dot(Omega, self.A)

                residual = pred-xt

            ffts = 0
            for j in range(xt.shape[1]):
                ffts += np.abs(np.fft.fft(residual[:, j])[:N//2])

            w = np.fft.fftfreq(N, 1)[:N//2]
            idxs = np.argmax(ffts)

            self.freqs.append(w[idxs])

            t = np.expand_dims(np.arange(N)+1, -1)

            Omega = np.concatenate([np.cos(t*2*np.pi*self.freqs), np.sin(t*2*np.pi*self.freqs)], -1)

            self.A = np.dot(np.linalg.pinv(Omega), xt)

    def sgd(self, xt, iterations=1000, learning_rate=3E-9, verbose=False):
        '''
        Given temporal data xt, sgd improves the initial guess of omega
        by SGD. It uses the pseudo-inverse to obtain A.

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data of dimensions [T, ...]
        iterations : TYPE int, optional
            Number of SGD iterations to perform. The default is 1000.
        learning_rate : TYPE float, optional
            Note that the learning rate should decrease with T. The default is 3E-9.
        verbose : TYPE, optional
            The default is False.

        Returns
        -------
        None.

        '''

        A = torch.tensor(self.A, requires_grad=False, device=self.device, dtype=torch.float32)
        freqs = torch.tensor(self.freqs, requires_grad=True, device=self.device, dtype=torch.float32)
        xt = torch.tensor(xt, requires_grad=False, device=self.device)

        o2 = torch.optim.SGD([freqs], lr=learning_rate)

        t = torch.unsqueeze(torch.arange(len(xt), dtype=torch.float32, device=self.device)+1, -1)

        loss = 0

        for i in range(iterations):

            Omega = torch.cat([torch.cos(t*2*np.pi*freqs), torch.sin(t*2*np.pi*freqs)], -1)
            A = torch.matmul(torch.pinverse(Omega.data), xt)

            xhat = torch.matmul(Omega, A)
            loss = torch.mean((xhat-xt)**2)

            o2.zero_grad()
            loss.backward()
            o2.step()

            loss = loss.cpu().detach().numpy()
            if verbose:
                print(loss)

        self.A = A.cpu().detach().numpy()
        self.freqs = freqs.cpu().detach().numpy()

    def fit(self, xt, learning_rate=1E-5, iterations=1000, verbose=False):
        '''

        Parameters
        ----------
        xt : TYPE numpy.array
            Temporal data of dimensions [T, ...]
        learning_rate : TYPE float, optional
            The default is 1E-5.
        iterations : TYPE int, optional
            DESCRIPTION. The default is 1000.
        verbose : TYPE, optional
            The default is False.

        Returns
        -------
        None.

        '''

        self.fft(xt)
        self.sgd(xt, iterations=iterations, learning_rate=learning_rate/xt.shape[0], verbose=verbose)

    def predict(self, T):
        '''
        Predicts the data from 1 to T.

        Parameters
        ----------
        T : TYPE int
            Prediction horizon

        Returns
        -------
        TYPE numpy.array
            xhat from 0 to T.

        '''

        t = np.expand_dims(np.arange(T)+1, -1)
        Omega = np.concatenate([np.cos(t*2*np.pi*self.freqs), np.sin(t*2*np.pi*self.freqs)], -1)

        return np.dot(Omega, self.A)


def fourier_loss(preds, trues, freqs, device='cpu'):
    # data shape: [B, T, D], freqs shape: [K], A shape: [2K, D]
    B, T, D = preds.shape
    preds = preds.reshape(-1, D)
    trues = trues.reshape(-1, D)

    t = torch.unsqueeze(torch.arange(len(preds), dtype=preds.dtype, device=device) + 1, -1)  # [B*T, 1]
    Omega = torch.cat([torch.cos(t*2*np.pi*freqs), torch.sin(t*2*np.pi*freqs)], -1)  # [B*T, 2K]

    Ap = torch.matmul(torch.pinverse(Omega.data), preds)  # [2K, D]
    preds_ = torch.matmul(Omega, Ap)  # [B*T, D]

    At = torch.matmul(torch.pinverse(Omega.data), trues)  # [2K, D]
    trues_ = torch.matmul(Omega, At)  # [B*T, D]

    loss = torch.mean((preds_-trues_)**2)
    return loss
