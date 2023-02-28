# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:17:21 2022

learnable comb filter

@author: Admin
"""
import numpy as np 
import torch.nn as nn
import torch
import torch.nn.functional as F

class comb_filter(nn.Module):
    def __init__(self, order = 3,
                 N_fft = 1536, 
                 N_hop = 384,
                 low_f0 = 62.5, 
                 high_f0 = 500,
                 stride = 3,
                 fs = 48000,
                 mode = 'linear',
                 trainable = True,
                 center = False,
                 **kwargs):
        super(comb_filter, self).__init__(**kwargs)

        self.N_fft = N_fft
        self.N_hop = N_hop
        self.frame_size = (N_fft // 2 + int(fs//low_f0) * (order - 1) // 2) * 2
        self.max_delay = self.frame_size // 2 - N_fft // 2
        self.low_f0 = low_f0
        self.high_f0 = high_f0
        
        self.center_freqs, self.filter_delays = self.get_freq_points(low_f0 = low_f0, 
                                                           high_f0 = high_f0,
                                                           stride = stride,
                                                           max_delay = self.max_delay,
                                                           fs = fs,
                                                           mode = mode)
        N_freqs = len(self.center_freqs)
        self.N_pitch = N_freqs + 1
        self.conv = torch.nn.Conv2d(1, N_freqs+1, [self.max_delay * 2 + 1, 1], [1,1], bias=False)
        self.conv.weight.requires_grad = trainable
        self.init_weights(self.conv, self.filter_delays, order)
        
        self.window = self.get_window(window = 'hanning')
        
        self.unfold = nn.Unfold(kernel_size=[self.frame_size,1], stride=[N_hop,1])
        self.pad = nn.ConstantPad1d([int(fs//low_f0) * (order - 1) // 2, int(fs//low_f0) * (order - 1) // 2],0)
        self.center = center
        if center:
            self.center_pad = nn.ReflectionPad1d([N_fft//2,N_fft//2])
            
    def forward_eval_single(self, x, pitch):
        '''
        accelerate the inference of a single audio
        x: (bs,T)
        pitch: (bs,T)
        '''
        if self.center:
            x = self.center_pad(x)
        x = self.pad(x)
        # (bs,1,T,1)
        x = x[:,None,:,None]
        # (bs, 3072, N_chunk)
        chunks = self.unfold(x)
        # (bs, 1, 3072, N_chunk)
        chunks = chunks[:,None]
        bs,_,L,T = chunks.shape
        
        # bs,T,N_pitch
        one_hot = F.one_hot(pitch.long(), num_classes=self.N_pitch).float()
        
        filtered_chunks = []
        for i in range(T):
            # 1, 1537
            filter_w = one_hot[:,i,:] @ self.conv.weight[:,0,:,0]
            # bs, 1, 1536, 1
            filtered_chunks.append(torch.conv2d(chunks[:,:,:,i:i+1], filter_w[:,None,:,None]))
        
        # (bs, T, 1536, 1) 
        filtered_chunks = torch.concat(filtered_chunks, dim = 1) * self.window[None,None,:,None].to(x.device)
        return filtered_chunks

    def forward(self, x, pitch):
        '''
        x: (bs,T)
        pitch: (bs,T)
        '''
        if self.center:
            x = self.center_pad(x)
        x = self.pad(x)
        # (bs,1,T,1)
        x = x[:,None,:,None]
        # (bs, 3072, N_chunk)
        chunks = self.unfold(x)
        # (bs, 1, 3072, N_chunk)
        chunks = chunks[:,None]
        bs,_,L,T = chunks.shape
        # (bs,N_pitch,1536,T)
        filtered_chunks = self.conv(chunks)
        # add window
        filtered_chunks = filtered_chunks * self.window[None,None,:,None].to(filtered_chunks.device)
        # (bs, T, 1536, N_picth) 
        filtered_chunks = filtered_chunks.permute([0,3,2,1])
        # bs,T,N_pitch,1
        one_hot = F.one_hot(pitch.long(), num_classes=self.N_pitch).float()[:,:,:,None]
        # bs,T,1536,1
        filtered_frame = filtered_chunks @ one_hot
        return filtered_frame
        
        
    def get_freq_points(self, low_f0, high_f0, stride = 3, max_delay = 768, fs = 48000, mode = 'linear'):
        '''
        Parameters
        ----------
        low_f0 : float
        high_f0 : float
        stride : int
            the minimum stride of the filters in the time domain. The default is 3.
        max_delay : int
            the max_delay of the filters. The default is 768.
        fs : int
            the sample rate. The default is 48000.
        mode : str
            linear intervals or logarithmic intervals. The default is 'linear'.
        Returns
        -------
        center_freqs：arr 
            the center freqs of the filters
        filter_delays：arr
            the delays of the filters
        '''
        assert mode in ['linear', 'logarithmic']
        min_freq = fs / max_delay
        delays = np.linspace(stride, max_delay, max_delay // stride)[::-1]
        freqs = fs / delays
        
        if mode == 'linear':
            low_f0_index = (np.abs(freqs - low_f0)).argmin()
            high_f0_index = (np.abs(freqs - high_f0)).argmin()
            center_freqs = freqs[low_f0_index:high_f0_index+1]
            filter_delays = delays[low_f0_index:high_f0_index+1].astype('int')
            
        elif mode == 'logarithmic':
            low_cent = self.hz2cent(hz=low_f0, low_f0=low_f0)
            high_cent = self.hz2cent(hz=high_f0, low_f0=low_f0)
            cents = np.linspace(low_cent, high_cent, 300)
            center_freqs = np.zeros(300)
            filter_delays = np.zeros(300, dtype = np.int)
            for i, cent in enumerate(cents):
                freq = self.cent2hz(cent)
                freq_index = (np.abs(freqs - freq)).argmin()
                center_freqs[i] = freqs[freq_index]
                filter_delays[i] = delays[freq_index]
            center_freqs = center_freqs[::-1]
            filter_delays = filter_delays[::-1]
        return center_freqs, filter_delays
    
    def init_weights(self, conv, filter_delays, order = 3):
        '''
        Initialized the weights of the Conv2d
        '''
        N_filter,_,kernel_size,_ = conv.weight.shape
        weights = np.zeros([N_filter, kernel_size], dtype = np.float32)
        
        filter_ws = {}
        for i in range(order, order + 8, 2):
            filter_ws[i] = self.get_comb_filter_weight(i)
    
        filter_w = self.get_comb_filter_weight(order)
        mid_index = kernel_size // 2
        for i in range(len(filter_delays)):
            weights[i,mid_index] = filter_w[order // 2]
            for j in range((order-1)//2):
                if mid_index + filter_delays[i] * (j+1) - 1 <= kernel_size - 1:
                    weights[i, mid_index + filter_delays[i] * (j+1) - 1] = filter_w[order // 2 + (j+1)]
                if mid_index - filter_delays[i] * (j+1) + 1 >= 0:
                    weights[i, mid_index - filter_delays[i] * (j+1) + 1] = filter_w[order // 2 - (j+1)]
        
        weights[-1,mid_index] = 1
        # normalization
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        # N_filter,1,kernel_size,1
        conv.weight.data = torch.from_numpy(weights[:,None,:,None])
    
    def get_comb_filter_weight(self, order):
        '''
        Parameters
        ----------
        order : int
        Returns
        -------
            The weights of the comb filter according to the order of the filter
        '''
        assert order % 2 == 1
        weight = np.hanning(order+2) #np.ones(order+2) 
        weight = weight / np.sum(weight)
        return weight[1:-1].astype('float32')
    
    def get_window(self, window):
        
        assert window in ['sine', 'hanning', 'hamming']
        if window == 'sine':
            return torch.sqrt(torch.hann_window(self.N_fft))
        elif window == 'hanning':
            return torch.hann_window(self.N_fft)
        elif window == 'hamming':
            return torch.hamming_window(self.N_fft)
    
    @staticmethod    
    def hz2cent(hz, low_f0 = 62.5, A = 100):
        '''
        map from 62.5hz - 500hz (3 oct.) to 300 cents
        '''
        return A * np.log2(hz / low_f0)
    
    @staticmethod
    def cent2hz(cent, low_f0 = 62.5, A = 100):
        '''
        map from 300 cents to 62.5hz - 500hz (3 oct.)
        '''
        return low_f0 * 2 ** (cent / A)    

def hz_to_points(freq_samples, pitch):
    for i,f in enumerate(pitch):
        if f > 0 :
            pitch[i] = (np.abs(freq_samples-f)).argmin()
    pitch[pitch<0] = 225
    return pitch

def get_pitch(s, center_freqs, frame_length=1536,hop_length=384):
    s = s / max(np.abs(s)+1e-8)
    # very slow
    f0 = librosa.pitch.pyin(s,62.5,500,48000,frame_length=frame_length,hop_length=hop_length,center=True)
    #f0 = librosa.pitch.pyin(librosa.resample(s,48000,16000),62.5,500,16000,frame_length=512,hop_length=128,center=True)
    pitch = f0[0]
    pitch[np.isnan(pitch)] = -225

    pitch = hz_to_points(center_freqs, pitch)
    return pitch.astype('int64')

def pitch_filter(f,out_f,cf):
    alpha = np.zeros(769)
    alpha[:80] = 1
    alpha[80:80+160] = np.hanning(320)[160:]
    
    s = sf.read(f)[0]

    f0 = librosa.pitch.pyin(s,62.5,500,48000,frame_length=1536,hop_length=384,center=True)
    pitch = f0[0]
    pitch[np.isnan(pitch)] = -225
    freq_samples = cf.center_freqs
    pitch = hz_to_points(freq_samples, pitch)

    #pitch to one hot
    pitch = torch.from_numpy(pitch[None]).long()
    s = s[None]
    with torch.no_grad():
        x = cf(torch.from_numpy(s.astype('float32')),pitch)
    
    spec = np.fft.rfft(x[0,:,:,0],axis=-1)
    
    spec0 = librosa.stft(s[0],1536,384,center=True)
    
    spec = spec.T * alpha[:,None]+ (1-alpha[:,None])*spec0
    
    combed_filtered_s = librosa.istft(spec,hop_length=384,center=True)

    s = s[0]

    sf.write(out_f,combed_filtered_s,48000)
    return s, combed_filtered_s

if __name__ == '__main__':
    import soundfile as sf
    import librosa
    np.random.seed(4)
    cf = comb_filter(mode = 'linear',center=True)

    s = sf.read('./A1DRKZ3SCLAS4V_M_Water_Near_Regular_SP_Mobile_Primary.wav')[0][:48000*10][None,].astype('float32')
    #s = np.random.randn(1,48000*2).astype('float32')
    #get f0
    f0 = librosa.pitch.pyin(s,62.5,500,48000,frame_length=1536,hop_length=384,center=True)
    pitch = f0[0][0]
    pitch[np.isnan(pitch)] = -225
    freq_samples = cf.center_freqs
    pitch = hz_to_points(freq_samples, pitch)
    #pitch to one hot
    pitch = torch.from_numpy(pitch[None]).long()
    s = s+np.random.randn(1,s.shape[-1]) * 0.0#1
    #c,r,i,r1,i1 = cf(torch.from_numpy(s.astype('float32')),pitch)
    x = cf(torch.from_numpy(s.astype('float32')),pitch)
    
    spec = np.fft.rfft(x[0,:,:,0],axis=-1)
    combed_filtered_s = librosa.istft(spec.T,hop_length=384,center=True)

    s = s[0]
    sf.write('./origin.wav',s,48000)
    sf.write('./comb_filtered.wav',combed_filtered_s,48000)
