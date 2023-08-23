import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def create_random_plot(fig_path,figsize=(6, 8),fontsize=12,save_figs=False,t=0,dt=500,nfig=0):
    ncol = 3
    fig = plt.figure(figsize=figsize) 
    gs = gridspec.GridSpec(nrows=4, ncols=3) 
    gs0 = gridspec.GridSpecFromSubplotSpec(1, ncol, subplot_spec=gs[0,:], wspace=.4,hspace=.5)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, ncol, subplot_spec=gs[1,:], wspace=.2,hspace=.5)
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2:,:], wspace=.2,hspace=.5)

    x_all = []
    for k in range(ncol):
        ax = plt.subplot(gs0[0,k])
        plot_x = (k+1)*np.random.randn(1000,ncol)
        x_all.append(plot_x)
        ax.plot(plot_x[:,0], plot_x[:,1], '-k', lw=1)
        ax.set_xlabel("$x_1$",fontsize=fontsize)
        ax.set_ylabel("$x_2$",fontsize=fontsize)
        ax.set_xticks([-1,0,1])
        ax.set_xticklabels([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_yticklabels([-1,0,1])
        ax.set_title(f'Latent States {k+1}' '\n' r'$\tau=${}'.format(k),fontsize=fontsize)
        ax.set_aspect('equal', 'box')
        
        ax = plt.subplot(gs1[0,k])
        ax.acorr(plot_x[:,k],maxlags=100,usevlines=True,linestyle='-',color='k',lw=2)
        ax.set_xlabel("Lag",fontsize=fontsize)
        ax.set_yticks([0,1])
        ax.set_ylim(0,1)
        if k == 0:
            ax.set_ylabel("Autocorrelation",fontsize=fontsize)
        elif k == ncol-1:
            ax.set_yticks([])
    x_all = np.concatenate(x_all,axis=1)
    ax = plt.subplot(gs2[0])
    for n in range(x_all.shape[-1]):
        ax.plot(x_all[t:t+dt,n]/np.max(np.abs(x_all[t:t+dt,n])) + 2*n,'-k')
    ax.set_ylabel("Latent States",fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0,2*x_all.shape[-1],2))
    ax.set_yticklabels(np.arange(1,x_all.shape[-1]+1))

    plt.tight_layout()

    if save_figs:
        fig.savefig(fig_path/'{}_RandomPlot.png'.format(nfig))
        nfig = nfig+1

    return nfig, fig


def plot_figures(config,fig_path,save_figs=True,t=0,dt=500):
    
    ##### Plotting ######
    nfig=0

    print('Plotting Results...')
    nfig,fig     = create_random_plot(fig_path,save_figs=save_figs,t=t,dt=dt,nfig=nfig)
    nfig,fig     = create_random_plot(fig_path,save_figs=save_figs,t=t,dt=dt,nfig=nfig)
    nfig,fig     = create_random_plot(fig_path,save_figs=save_figs,t=t,dt=dt,nfig=nfig)