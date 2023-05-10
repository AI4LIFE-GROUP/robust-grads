import matplotlib.pyplot as plt
import numpy as np
from joblib import load

def weight_grad_plot(thresholds, weight_change, grad_change, labs, qs=[0.25, 0.5, 0.75]):
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(8,3))
    plt.subplots_adjust(wspace=0.3)
    ylabs = ['$||\\theta_1-\\theta_2||_2$' , '$||g_1-g_2||_2$']
    for i in range(weight_change.shape[0]):
        if weight_change.shape[1] == 1:
            ax[0].plot(thresholds, weight_change[i,0], label=labs[i])
            ax[1].plot(thresholds, np.mean(grad_change[i,0], axis=1), label=labs[i])
        else:
            q = np.quantile(weight_change[i], qs, axis=0)
            ax[0].plot(thresholds, q[1], label=labs[i])
            ax[0].fill_between(thresholds, q[0], q[2], alpha=0.4)
            q = np.quantile(np.mean(grad_change[i], axis=2), qs, axis=0)
            ax[1].plot(thresholds, q[1], label=labs[i])
            ax[1].fill_between(thresholds, q[0], q[2], alpha=0.4)
    ax[0].set_title('Change in Parameters')
    ax[1].set_title('Mean Change in Gradients')
    for i in range(2):
        ax[i].set_ylabel(ylabs[i])
        ax[i].set_xlabel('Standard Deviation of Noise, $\sigma$')
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
    ax[0].legend(ncol=4, bbox_to_anchor=(1.95,-0.2))
    plt.show()
    
    
def weight_plot(thresholds, weight_change1, weight_change2, labs, dnames, qs=[0.25, 0.5, 0.75]):
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(8,3))
    plt.subplots_adjust(wspace=0.3)
    ylab = '$||\\theta_1-\\theta_2||_2$'
    
    for i in range(weight_change1.shape[0]):
        q = np.quantile(weight_change1[i], qs, axis=0)
        ax[0].plot(thresholds, q[1], label=labs[i])
        ax[0].fill_between(thresholds, q[0], q[2], alpha=0.4)
        q = np.quantile(weight_change2[i], qs, axis=0)
        ax[1].plot(thresholds, q[1], label=labs[i])
        ax[1].fill_between(thresholds, q[0], q[2], alpha=0.4)
    for i in range(2):
        ax[i].set_title(f'{dnames[i]}\nChange in Parameters')
        ax[i].set_ylabel(ylab)
        ax[i].set_xlabel('Standard Deviation of Noise, $\sigma$')
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
    ax[0].legend(ncol=4, bbox_to_anchor=(2,-0.2))
    plt.show()
    
def grad_plot(thresholds, grad_change1, grad_change2, labs, dnames, qs=[0.25, 0.5, 0.75]):
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(8,3))
    plt.subplots_adjust(wspace=0.3)
    ylab = '$||g_1-g_2||_2$'
    
    for i in range(grad_change1.shape[0]):
        q = np.quantile(np.mean(grad_change1[i], axis=2), qs, axis=0)
        ax[0].plot(thresholds, q[1], label=labs[i])
        ax[0].fill_between(thresholds, q[0], q[2], alpha=0.4)
        q = np.quantile(np.mean(grad_change2[i], axis=2), qs, axis=0)
        ax[1].plot(thresholds, q[1], label=labs[i])
        ax[1].fill_between(thresholds, q[0], q[2], alpha=0.4)
    for i in range(2):
        ax[i].set_title(f'{dnames[i]}\nMean Change in Gradients')
        ax[i].set_ylabel(ylab)
        ax[i].set_xlabel('Standard Deviation of Noise, $\sigma$')
        ax[i].set_yscale('log')
        ax[i].set_xscale('log')
    ax[0].legend(ncol=4, bbox_to_anchor=(2,-0.2))
    plt.show()
    
    
def top_k_plot(thresholds, exps, exp_names, met_name, labs,
               output_dir, dname, dname_y=0.66, k=5, qs=[0.25, 0.5, 0.75]):
    
    fig, ax = plt.subplots(1, 4, figsize=(17, 3), dpi=200)
    fig.suptitle(dname, fontsize=18, fontweight='bold', x=0.075, y=dname_y, rotation='vertical')
    plt.subplots_adjust(wspace=0.4)

    ax[0].text(1.18, 1.13, 'Weight Decay', fontsize=15, fontweight='bold', transform=ax[0].transAxes, ha='center')
    ax[2].text(1.19, 1.13, 'Curvature', fontsize=15, fontweight='bold', transform=ax[2].transAxes, ha='center')

    for col in range(4):
        exp = exps[col%2]
        var_name = '' if col//2 else '_wd'
        exp_name = exp_names[col%2]
        topk = np.load(output_dir+f'/{exp}_top_{k}_{met_name}{var_name}.npy')
        for i, lab in enumerate(labs[col//2]):
            if topk.shape[1] == 1:
                ax[col].plot(thresholds, topk[i,0], label=lab)
            else:
                q = np.quantile(topk[i], qs, axis=0)
                ax[col].plot(thresholds, q[1], label=lab)
                ax[col].fill_between(thresholds, q[0], q[2], alpha=0.4)
        if col == 0:
            ax[col].legend(ncol=4, bbox_to_anchor=(2.15, -0.2))
        elif col == 2:
            ax[col].legend(ncol=3, bbox_to_anchor=(2.05, -0.2))
        ax[col].set_xlabel('Standard Deviation of Noise, $\sigma$')
        ax[col].set_xscale('log')
        ax[col].set_title(exp_name)
        ax[col].set_ylabel(f'Top-{k} {met_name.upper()}')
    plt.show()
    
    
def top_k_plot2(thresholds, exps, exp_names, met_name, labs,
                output_dir, k=5, qs=[0.25, 0.5, 0.75]):
    fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200, figsize=(8,3))
    plt.subplots_adjust(wspace=0.3)
    ylab = f'Top-{k} {met_name.upper()}'
    var_name = '' if len(labs)==3 else '_wd'
    
    for col in range(2):
        topk = np.load(output_dir+f'/{exps[col]}_top_{k}_{met_name}{var_name}.npy')
        for i, lab in enumerate(labs):
            q = np.quantile(topk[i], qs, axis=0)
            ax[col].plot(thresholds, q[1], label=lab)
            ax[col].fill_between(thresholds, q[0], q[2], alpha=0.4)
        ax[col].set_title(exp_names[col])
        ax[col].set_ylabel(ylab)
        ax[col].set_xlabel('Standard Deviation of Noise, $\sigma$')
        ax[col].set_xscale('log')
    ax[0].legend(ncol=4, bbox_to_anchor=(2,-0.2))
    plt.show()


def top_k_plot_tree(thresholds, exps, exp_names, met_name, labs,
                    output_dir, dname, dname_y=0.66, k=5, qs=[0.25, 0.5, 0.75]):
    
    fig, ax = plt.subplots(1, 4, figsize=(17, 3), dpi=200)
    fig.suptitle(dname, fontsize=18, fontweight='bold', x=0.075, y=dname_y, rotation='vertical')
    plt.subplots_adjust(wspace=0.4)

    ax[0].text(1.18, 1.13, 'Depth', fontsize=15, fontweight='bold', transform=ax[0].transAxes, ha='center')
    ax[2].text(1.19, 1.13, 'Depth', fontsize=15, fontweight='bold', transform=ax[2].transAxes, ha='center')

    for col in range(4):
        exp = exps[col%2]
        var_name = '' if col//2 else '_wd'
        exp_name = exp_names[col%2]
        topk = load(output_dir+f'/{exp}_top_{k}_{met_name}{var_name}.npy')
        for i, lab in enumerate(labs[col//2]):
            if topk.shape[1] == 1:
                ax[col].plot(thresholds, topk[i,0], label=lab)
            else:
                q = np.quantile(topk[i], qs, axis=0)
                ax[col].plot(thresholds, q[1], label=lab)
                ax[col].fill_between(thresholds, q[0], q[2], alpha=0.4)
        if col == 0:
            ax[col].legend(ncol=4, bbox_to_anchor=(2.15, -0.2))
        elif col == 2:
            ax[col].legend(ncol=3, bbox_to_anchor=(2.05, -0.2))
        ax[col].set_xlabel('Standard Deviation of Noise, $\sigma$')
        ax[col].set_xscale('log')
        ax[col].set_title(exp_name)
        ax[col].set_ylabel(f'Top-{k} {met_name.upper()}')
    plt.show()