# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:29:14 2022

@author: Admin
"""

import torch 
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as functional
import python_speech_features as psf

from modules import DprnnBlock#


#%% causal Conv2d Deconv2d
class DilatConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 dilation: tuple=(1,1),
                 groups: int=1,
                 causal: bool=True):
        """
        causal dilated convlutional layer:
        If the kernel size of the time axis over 1, a causal padding will be applied before the convlution.
        """
        super(DilatConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        k_t,k_f = kernel_size
        d_t,d_f = dilation
        
        k_t = k_t + (k_t-1) * (d_t-1)
        k_f = k_f + (k_f-1) * (d_f-1)
            
        if k_t > 1:
            # freq same padding
            freq_padding_start = (k_f - 1)//2
            freq_padding_end = k_f - 1 - (k_f -1)//2
            if causal:
                # time causal padding
                padding = (freq_padding_start, freq_padding_end, k_t-1, 0)
            else:
                # look ahead 1 frame
                padding = (freq_padding_start, freq_padding_end, k_t-2, 1)
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding='same', groups=groups)
            
    def forward(self, inputs: Tensor) -> Tensor:
        
        return self.conv(inputs)

class DilatDeConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 slices: tuple,
                 dilation: tuple=(1,1),
                 groups: int=1):     
            
        super(DilatDeConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.k_t, self.k_f = kernel_size
        self.d_t, self.d_f = dilation
        self.s_t, self.s_f = stride        
        
        self.k_t = self.k_t + (self.k_t-1) * (self.d_t-1)
        self.k_f = self.k_f + (self.k_f-1) * (self.d_f-1)
        self.slices = slices

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups)

    def forward(self, inputs: Tensor) -> Tensor:
        
        x = self.conv(inputs)
        if self.k_t > 1:
            if self.s_f == 1:
                x = x[:,:,:-(self.k_t-1),1:-1]
            elif self.s_f == 2:
                x = x[:,:,:-(self.k_t-1),self.slices[0]:self.slices[1]] 
        return x

#%% DepthSepConv2d DepthSep deconv2d
class DepthSepConv2d(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride,
                 dilation=(1,1),
                 groups=None,
                 normal=False,
                 causal=True):
        super(DepthSepConv2d, self).__init__()
        self.normal = normal
        if not normal:
            if groups == None:
                groups = out_channels
            self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.depth_conv = DilatConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, causal=causal)
        else:
            if groups == None:
                groups = in_channels
            self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.depth_conv = DilatConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, causal=causal)
            

    def forward(self, x):
        if not self.normal:
            return self.depth_conv(self.point_conv(x))
        else:
            return self.point_conv(self.depth_conv(x))

class DepthSepDeConv2d(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride,
                 slices,
                 dilation=(1,1),
                 groups=None,
                 normal=False):
        super(DepthSepDeConv2d, self).__init__()
        self.normal = normal
        if not normal:
            if groups == None:
                groups = out_channels
                
            self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.depth_deconv = DilatDeConv2d(in_channels=out_channels , out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, slices=slices)
        else:
            if groups == None:
                groups = in_channels
            self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.depth_deconv = DilatDeConv2d(in_channels=in_channels , out_channels=in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=groups, slices=slices)
         
    def forward(self, x):
        if not self.normal:
            return self.depth_deconv(self.point_conv(x))
        else:
            return self.point_conv(self.depth_deconv(x))
        
#%% gateconv2d
class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: list,
                 dilation: tuple =(1,1),
                 gate: bool = True,
                 depth_sep: bool = False,
                 causal: bool = True):
        """
        Causal convlutional layer:
        If the kernel size of the time axis over 1, a causal padding will be applied before the convlution.
        """
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.gate = gate
        self.depth_sep = depth_sep
        k_t = kernel_size[0]

        if gate:
            out_channels = out_channels * 2
            
        if depth_sep:
            Conv2d = DepthSepConv2d
            self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal = causal)

        else:
            Conv2d = nn.Conv2d
            padding = (padding[0], padding[1], k_t-1, 0)
            self.conv = nn.Sequential(
                    nn.ConstantPad2d(padding, value=0.),
                    Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        if self.gate:
            outputs, gate = x.chunk(2, dim=1)
            return outputs * gate.sigmoid()
        else:
            return x
        
class GateDeConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 slices: tuple,
                 dilation: tuple =(1,1),
                 gate: bool = True,
                 depth_sep: bool = False):
        """
        Causal convlutional layer:
        If the kernel size of the time axis over 1, a causal slicing will be applied after the convlution.
        A symmtric slicing is appiled at the frequency axis.
        """
        super(GateDeConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.gate = gate
        self.depth_sep = depth_sep
        self.k_t, self.k_f = kernel_size
        self.s_t, self.s_f = stride
        self.slices = slices
        
        if depth_sep:
            DeConv2d = DepthSepDeConv2d
        else:
            pass
            
        if gate:
            out_channels = out_channels * 2
            
        self.conv = DeConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, slices=slices)

    def forward(self, inputs: Tensor) -> Tensor:
        
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        
        if self.gate:
            outputs, gate = x.chunk(2, dim=1)
            return outputs * gate.sigmoid()
        else:
            return x
          
#%%
class Encoder(nn.Module):
    def __init__(self, in_channel, 
                       filter_size, 
                       kernel_size, 
                       strides, 
                       dilations, 
                       paddings, 
                       gate=False, 
                       depth_sep=False,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        """
        input: bs,C,T,F
        """
        self.N_layer = len(filter_size)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        self.paddings = paddings
        
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.padding_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.gate = gate
        
        for i in range(self.N_layer):
            if i == 0:
                self.conv_list.append(
                    GateConv2d(in_channels=in_channel,
                               out_channels=self.filter_size[i],
                               kernel_size=self.kernel_size[i],
                               stride=self.strides[i],
                               dilation=self.dilations[i],
                               padding=paddings[i],
                               gate=gate,
                               depth_sep=depth_sep,
                               causal= False))
            else:
                self.conv_list.append(
                    GateConv2d(in_channels=self.filter_size[i-1],
                               out_channels=self.filter_size[i],
                               kernel_size=self.kernel_size[i],
                               stride=self.strides[i],
                               dilation=self.dilations[i],
                               padding=paddings[i],
                               gate=gate,
                               depth_sep=depth_sep))
                
            self.bn_list.append(nn.BatchNorm2d(self.filter_size[i]))
            self.activation_list.append(nn.PReLU(self.filter_size[i]))
        
    def forward(self, x):
        
        encoder_out_list = []
        for i in range(self.N_layer):
            x = self.conv_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            encoder_out_list.append(x)
        return x,encoder_out_list

class Decoder(nn.Module):
    def __init__(self, out_channel, 
                       filter_size, 
                       kernel_size, 
                       strides, 
                       slices, 
                       dilations,
                       skip_connect = 'cat', 
                       gate=False, 
                       depth_sep=False, 
                       **kwargs):
        super(Decoder, self).__init__(**kwargs)
        """
        input: bs,C,T,F
        """
        self.N_layer = len(filter_size)
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.slices = slices
        self.out_channel = out_channel
        self.skip_connect = skip_connect
        self.gate = gate
        
        for i in range(self.N_layer):
            if skip_connect == 'cat':
                in_channels = self.filter_size[self.N_layer-1-i] * 2
            elif skip_connect == 'add':
                in_channels = self.filter_size[self.N_layer-1-i]
                
            if i == self.N_layer - 1:
                if out_channel == 2: # only crm
                    
                    self.deconv_crm = GateDeConv2d(in_channels=in_channels,
                                                 out_channels=out_channel,
                                                 kernel_size=self.kernel_size[self.N_layer-1-i],
                                                 stride=self.strides[self.N_layer-1-i],
                                                 dilation = self.dilations[self.N_layer-1-i],
                                                 slices=self.slices[self.N_layer-1-i],
                                                 gate=gate,
                                                 depth_sep=depth_sep)
                    self.bn_list.append(nn.BatchNorm2d(out_channel))
                elif out_channel == 3:# mask for mag and mask for phase
                    self.deconv_crm = GateDeConv2d(in_channels=in_channels,
                                                out_channels=2,
                                                kernel_size=self.kernel_size[self.N_layer-1-i],
                                                stride=self.strides[self.N_layer-1-i],
                                                dilation = self.dilations[self.N_layer-1-i],
                                                slices=self.slices[self.N_layer-1-i],
                                                gate=gate,
                                                depth_sep=depth_sep)   
                    
                    self.deconv_irm = GateDeConv2d(in_channels=in_channels,
                                        out_channels=1,
                                        kernel_size=self.kernel_size[self.N_layer-1-i],
                                        stride=self.strides[self.N_layer-1-i],
                                        dilation = self.dilations[self.N_layer-1-i],
                                        slices=self.slices[self.N_layer-1-i],
                                        gate=gate,
                                        depth_sep=depth_sep)   
                    
                    self.bn_crm = nn.BatchNorm2d(2)
                    self.bn_irm = nn.BatchNorm2d(1)
                else:
                    self.deconv_crm = GateDeConv2d(in_channels=in_channels,
                                        out_channels=out_channel,
                                        kernel_size=self.kernel_size[self.N_layer-1-i],
                                        stride=self.strides[self.N_layer-1-i],
                                        dilation = self.dilations[self.N_layer-1-i],
                                        slices=self.slices[self.N_layer-1-i],
                                        gate=gate,
                                        depth_sep=depth_sep)
            else:
                self.conv_list.append(
                    GateDeConv2d(in_channels=in_channels,
                              out_channels=self.filter_size[self.N_layer-2-i],
                              kernel_size=self.kernel_size[self.N_layer-1-i],
                              stride=self.strides[self.N_layer-1-i],
                              dilation = self.dilations[self.N_layer-1-i],
                              slices=self.slices[self.N_layer-1-i],
                              gate=gate,
                              depth_sep=depth_sep))
                
                self.activation_list.append(nn.PReLU(self.filter_size[self.N_layer-2-i]))       
                self.bn_list.append(nn.BatchNorm2d(self.filter_size[self.N_layer-2-i]))
            #self.padding_list.append(nn.ZeroPad2d([0,0,paddings[i][0],paddings[i][1]]))

    def forward(self, dp_in, encoder_out_list):
        
        if self.skip_connect == 'cat':
            x = torch.cat([dp_in, encoder_out_list[-1]], dim = 1)
        elif self.skip_connect == 'add':
            x = dp_in + encoder_out_list[-1]
            
        for i in range(self.N_layer-1):
            # bs, C, T, B
            x = self.conv_list[i](x)
           # print(x.shape)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            
            #print(x.shape)
            if self.skip_connect == 'cat':
                x = torch.cat([x, encoder_out_list[self.N_layer-2-i]], dim = 1)
            elif self.skip_connect == 'add':
                x = x + encoder_out_list[self.N_layer-2-i]
            
        out = self.deconv_crm(x)
        if self.out_channel == 2:
            out = self.bn_list[-1](out)
            return out
        
        elif self.out_channel == 3:
            out = self.bn_crm(out)
            mag = self.deconv_irm(x)
            mag = self.bn_irm(mag)
            return [out,mag]
        else:
            return out
        
class Mask(nn.Module):
    
    def __init__(self,df_acti = None):
        super(Mask, self).__init__()
        self.unfold = nn.Unfold(kernel_size=(1,3), padding=(0,1))
        self.mag_acti = nn.Sigmoid()
        self.df_acti = df_acti

    def forward(self, mask, spec):
        # mask.shape = [bs, 6, T, F], dtype=real
        # spec.shape = [bs, 2, T, F], dtype=complex
        if self.df_acti:
            mask_s1 = self.df_acti(mask[:,:3,:,:])
        else:
            mask_s1 = mask[:,:3,:,:]

        mask_s2_mag = self.mag_acti(mask[:,3,:,:])

        mask_s2_angle = torch.atan2(mask[:,5,:,:], mask[:,4,:,:]+1e-8)
        
        mag = torch.norm(spec, dim=1, keepdim=True)
        angle = torch.atan2(spec[:,-1,:,:], spec[:,0,:,:]+1e-8)
    
        mag_unfold = self.unfold(mag).reshape([spec.shape[0], 3, -1, spec.shape[3]])
        # bs,T,F
        x = torch.sum(mag_unfold * mask_s1, dim=1)
        
        real = x * mask_s2_mag * torch.cos(angle + mask_s2_angle)
        imag = x * mask_s2_mag * torch.sin(angle + mask_s2_angle)
        # bs,2,T,F
        return  torch.stack([real,imag], dim = 1)

class DPCRN(nn.Module):
    def __init__(self, in_channel, 
                       out_channel, 
                       filter_size, 
                       kernel_size, 
                       strides, 
                       dilations,
                       conv_paddings, 
                       deconv_slices, 
                       N_dprnn, 
                       intra_hidden, 
                       inter_hidden, 
                       batch_size, 
                       norm ='BN', 
                       gate = False, 
                       depth_sep = False,
                       causal = False,
                       long_Bi_GRU = False,
                       trainable_scm = False,
                       df_acti = None,
                       **kwargs):
        super(DPCRN, self).__init__(**kwargs)
        
        self.norm = norm
        if norm == 'BN':
            self.input_norm = nn.BatchNorm2d(in_channel, eps = 1e-8)
        elif norm == 'iLN':
            self.input_norm = nn.LayerNorm([in_channel, 256], eps = 1e-8) 
        
        m = psf.get_filterbanks(nfilt=80,nfft=1536,samplerate=48000).astype('float32')
        self.mel_mat = torch.from_numpy(m.T)

        self.encoder = Encoder(in_channel = in_channel,
                               filter_size = filter_size,
                               kernel_size = kernel_size, 
                               strides = strides, 
                               dilations = dilations,
                               paddings = conv_paddings,
                               gate = gate,
                               depth_sep = depth_sep)
        
        self.decoder = Decoder(out_channel = out_channel,
                               filter_size = filter_size, 
                               kernel_size = kernel_size,
                               strides = strides,
                               dilations = dilations,
                               slices = deconv_slices,
                               gate = gate,
                               depth_sep = depth_sep)

        self.decoder_r = Decoder(out_channel = out_channel,
                               filter_size = filter_size, 
                               kernel_size = kernel_size,
                               strides = strides,
                               dilations = dilations,
                               slices = deconv_slices,
                               gate = gate,
                               depth_sep = depth_sep)
        
        self.dprnns = nn.ModuleList()
        
        self.N_dprnn = N_dprnn
        
        self.linear_feature1 = nn.Sequential(nn.Linear(640,128,),nn.Sigmoid())
        self.linear_vad = nn.Sequential(nn.Linear(128,1),nn.Sigmoid())
        
        self.GRU_f0 = nn.GRU(192, 128, batch_first=True)
        self.linear_f0 = nn.Sequential(nn.Linear(128,226),nn.Sigmoid())
        
        for i in range(N_dprnn):

            self.dprnns.append(DprnnBlock(intra_hidden=intra_hidden,
                                          inter_hidden=inter_hidden,
                                          batch_size=batch_size, 
                                          L=-1, 
                                          width=10,
                                          channel=filter_size[-1], 
                                          causal = True,
                                          long_GRU = long_Bi_GRU))
    
    def forward(self, inp_spec, h_in):
        """
        x: bs,2,T,F
        h: 2,32,128
        """
        x = inp_spec
        self.mel_mat = self.mel_mat.to(x.device)
        h_out_list = []
        # bs,1,T,F
        x = inp_spec[:,0:1]**2 + inp_spec[:,1:2]**2
        x_low = torch.log(x[:,0,:,:64] + 1e-8)
        # bs,1,T,80
        x = x @ self.mel_mat
        x = torch.log(x + 1e-8)
        
        if self.norm == 'BN':
            x = self.input_norm(x)
        elif self.norm == 'iLN':
            x = self.input_norm(x.permute([0,2,1,3])).permute([0,2,1,3])
        
        dp_in, encoder_out_list = self.encoder(x)
        #[print(i.shape) for i in encoder_out_list]
        for i in range(self.N_dprnn):
            dp_in, h_out = self.dprnns[i](dp_in, h_in[i:i+1,:,:])
            h_out_list.append(h_out)
        
        bs,C,T,F = dp_in.shape
        feature1 = dp_in.permute([0,2,1,3]).reshape([bs,T,F*C])
        # bs,T,128
        feature2 = self.linear_feature1(feature1)
        esti_vad = self.linear_vad(feature2)
        # bs,T,192
        #feature3 = feature2
        feature3 = torch.concat([feature2,x_low], dim=-1)
        feature4,_ = self.GRU_f0(feature3)
        # bs,T,226
        esti_f0 = self.linear_f0(feature4)
        # 
        r = torch.nn.Sigmoid()(self.decoder_r(dp_in, encoder_out_list))
        output = torch.nn.Sigmoid()(self.decoder(dp_in, encoder_out_list))
        
        mask = output @ self.mel_mat.T
        r = r @ self.mel_mat.T
        enh_spec = mask * inp_spec
        
        return enh_spec, mask, esti_vad, esti_f0, r, h_out_list[0], h_out_list[1]
    
if __name__ == '__main__':

    # baseline
    dpcrn_args_dict = {"in_channel": 1,
                       "out_channel": 1,
                       "filter_size": [32,32,32,64,64],
                       "kernel_size": [(2,5),(2,3),(2,3),(2,3),(2,3)],
                       "strides": [[1,2],[1,2],[1,2],[1,1],[1,1]],
                       "dilations": [[1,1],[1,1],[1,1],[1,1],[1,1]],
                       "conv_paddings":[[1,2],[1,1],[1,1],[1,1],[1,1]],
                       "deconv_slices": [[1,-2],[0,-1],[0,-1],[1,-1],[1,-1]],
                       "N_dprnn": 2,
                       "intra_hidden": 64,
                       "inter_hidden": 64,
                       "batch_size": 1,
                       "gate": True,
                       "depth_sep": True}
    import soundfile as sf
    import librosa
    s = sf.read('./p232_010.wav')[0]
    spec = librosa.stft(s,1536,384)
    input_real = spec.real.T
    input_real = input_real[None,None]
    input_imag = spec.imag.T
    input_imag = input_imag[None,None]    
    
    dpcrn = DPCRN(**dpcrn_args_dict)
    from torchsummaryX import summary
    h_in = torch.zeros([2, 10, 64])
    input_feature = torch.from_numpy(np.concatenate([input_real,input_imag],axis=1).astype('float32'))
    enh_spec, mask, esti_vad, esti_f0, r, h0, h1 = dpcrn(input_feature, h_in)
    summary(dpcrn, input_feature, h_in)
    
