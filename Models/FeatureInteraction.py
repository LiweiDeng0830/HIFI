import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nvl,vw->nwl',(x,A))
        return x.contiguous()

class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.alpha = alpha

    def forward(self,x,adj):        # batch_size, seq_len, num_nodes
        x = x.transpose(1,2)        # batch_size, num_nodes, seq_len
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)   # batch_size, num_nodes*(1+gdep), seq_len
        ho = ho.transpose(1,2)      # batch_size, seq_len, num_nodes*(1+gdep)
        ho = self.mlp(ho)           # batch_size, seq_len, c_out
        return ho

class graph_constructor(nn.Module):
    def __init__(self, nnodes, dim, alpha=3, static_feat=None, k=2):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class graph_constructor_symetric(nn.Module):
    def __init__(self, nnodes, dim, alpha=3, static_feat=None, k=5):
        super(graph_constructor_symetric, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin = nn.Linear(xd, dim)
        else:
            self.emb = nn.Embedding(nnodes, dim)
            self.lin = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec = self.emb(idx)
        else:
            nodevec = self.static_feat[idx,:]

        nodevec = torch.tanh(self.alpha*self.lin(nodevec))

        a = torch.mm(nodevec, nodevec.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(idx.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec = self.emb(idx)
        else:
            nodevec = self.static_feat[idx,:]

        nodevec = torch.tanh(self.alpha*self.lin(nodevec))

        a = torch.mm(nodevec, nodevec.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj
