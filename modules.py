import math
import torch 
from torch import nn
import numpy as np
import torch.nn.functional as functional

class DprnnBlock(nn.Module):
    '''
    DPRNN block
    '''
    def __init__(self,  intra_hidden, inter_hidden, batch_size, L, width, channel, causal = True, batch_first = True, long_GRU = False,**kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        
        self.intra_hidden = intra_hidden
        self.inter_hidden = inter_hidden
        self.batch_size = batch_size
        self.L = L
        self.width = width
        self.channel = channel
        self.causal = causal 
        self.batch_first = batch_first
        long_GRU = long_GRU
        self.intra_rnn = nn.GRU(self.channel, self.intra_hidden//2, bidirectional=True, batch_first=batch_first)
        self.intra_fc = nn.Linear(self.intra_hidden, self.channel)
        self.intra_ln = nn.LayerNorm([self.channel], eps = 1e-8) # [self.width, self.channel]

        self.inter_rnn = nn.GRU(self.channel, self.inter_hidden, batch_first=batch_first)
        self.inter_fc = nn.Linear(self.inter_hidden, self.channel)
        self.inter_ln = nn.LayerNorm([self.channel], eps = 1e-8) # [self.width, self.channel]

    def init_hidden(self, bs, hs, device, nums_GRU = 1, **kwargs):
        init_h = torch.zeros(nums_GRU, bs, hs).to(device)
        return init_h
    
    def forward(self, x, h = None):
        
        if self.batch_first:
        # input shape (bs,C,T,F) --> (bs,T,F,C) --> (bs*T,F,C)
            intra_in = torch.permute(x, [0, 2, 3, 1])
            intra_rnn_in = torch.reshape(intra_in, [-1, self.width, self.channel])
        else:
        # input shape (bs,C,T,F) --> (F,bs,T,C) --> (F,bs*T,C)
            pass
        
        # (bs*T,F,C) / (F,bs*T,C)
        intra_rnn_out, _ = self.intra_rnn(intra_rnn_in)
        # (bs*T,F,C) / (F,bs*T,C)
        intra_fc_out = self.intra_fc(intra_rnn_out)
        
        if self.batch_first:
            if self.causal:
                # (bs*T,F,C) --> (bs,T,F,C)
                intra_LN_in = torch.reshape(intra_fc_out, [self.batch_size, -1, self.width, self.channel])
                intra_out = self.intra_ln(intra_LN_in)
            else:       
                # (bs*T,F,C) --> (bs,T*F,C) --> (bs,T,F,C)
                intra_LN_in = torch.reshape(intra_fc_out, [self.batch_size, -1, self.channel])
                intra_LN_out = self.intra_ln(intra_LN_in)
                intra_out = torch.reshape(intra_LN_out, [self.batch_size, -1, self.width, self.channel])
        else:
            pass    
        # (bs,T,F,C) 
        intra_out = intra_in + intra_out
        
        #%%
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_rnn_in = torch.permute(intra_out, [0, 2, 1, 3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_rnn_in = torch.reshape(inter_rnn_in, [self.batch_size * self.width, -1, self.channel])
        
        # (bs*F,T,C) / (T,bs*F,C) h:(1,bs*F,C)
        if h is None:
            inter_rnn_out, h_inter = self.inter_rnn(inter_rnn_in)
        else:
            inter_rnn_out, h_inter = self.inter_rnn(inter_rnn_in, h)
        
        # (bs*F,T,C) / (T,bs*F,C)
        inter_fc_out = self.inter_fc(inter_rnn_out)
        
        if self.batch_first:
            if self.causal:
                # (bs*F,T,C) --> (bs,F,T,C) --> (bs,T,F,C)
                inter_LN_in = torch.reshape(inter_fc_out, [self.batch_size, self.width, -1, self.channel])
                inter_LN_in = torch.permute(inter_LN_in, [0,2,1,3])
                inter_out = self.inter_ln(inter_LN_in)
            else:
                # (bs*F,T,C) --> (bs,F*T,C) --> (bs,F,T,C) --> (bs,T,F,C)
                inter_LN_in = torch.reshape(inter_fc_out, [self.batch_size, -1, self.channel])
                inter_LN_out = self.inter_ln(inter_LN_in)
                inter_out = torch.reshape(inter_LN_out, [self.batch_size, self.width, -1, self.channel])
                inter_out = torch.permute(inter_out, [0,2,1,3])
        else:
            pass
        # (bs,T,F,C)            
        inter_out = intra_out + inter_out
        # (bs,C,T,F)
        inter_out = torch.permute(inter_out, [0, 3, 1, 2])
        return inter_out, h_inter
