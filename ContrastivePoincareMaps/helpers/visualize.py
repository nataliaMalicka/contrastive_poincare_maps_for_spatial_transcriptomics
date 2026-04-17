"""
Visualisation helpers. Implementation adapted from: https://github.com/facebookresearch/PoincareMaps
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import random
import textwrap

warnings.filterwarnings("ignore")
plt.switch_backend('agg')
sns.set()


colors_palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B',
                  '#762A83', '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7',
                  '#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B',
                  '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
                  '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
                  '#CCEBC5', '#FFED6F', '#edf8b1', '#c7e9b4', '#7fcdbb',
                  '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58']

def linear_scale(embeddings):
    embeddings = np.transpose(embeddings)
    sqnorm = np.sum(embeddings ** 2, axis=1, keepdims=True)
    dist = np.arccosh(1 + 2 * sqnorm / (1 - sqnorm))
    dist = np.sqrt(dist)
    dist /= dist.max()
    sqnorm[sqnorm == 0] = 1
    embeddings = dist * embeddings / np.sqrt(sqnorm)
    return np.transpose(embeddings)

def plot_poincare_disc(x, labels=None, labels_name='labels', labels_order=None, 
                       file_name=None, coldict=None,
                       d1=19, d2=18.0, fs=11, ms=5, col_palette=plt.get_cmap("tab10"), bbox=(1.3, 0.7)):

    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])
    
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)

    if not (labels is None):
        df[labels_name] = labels[idx]
        if labels_order is None:
            labels_order = np.unique(labels)        
        if coldict is None:
            coldict = dict(zip(labels_order, col_palette[:len(labels)]))
        sns.scatterplot(x="pm1", y="pm2", hue=labels_name, 
                        hue_order=labels_order,
                        palette=coldict,
                        data=df, ax=ax, s=ms)

        #ax.legend(fontsize=fs, loc='best', bbox_to_anchor=bbox)
        print("plotting legend")
        handles, leg_labels = ax.get_legend_handles_labels()
        if labels_name in leg_labels:
            idx_title = leg_labels.index(labels_name)
            handles.pop(idx_title)
            leg_labels.pop(idx_title)

        # Remove the legend from the main plot
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        # ---- Create a separate legend figure ----
        n_items = len(leg_labels)
        ncols = (n_items + 1) // 15  # two rows

        legend_fig = plt.figure(figsize=(ncols * 1.2, 1.6))  # adjust size to fit
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")


        legend = legend_ax.legend(
            handles, leg_labels,
            loc="center",
            ncol=ncols,
            frameon=False,
            fontsize=12,
            markerscale=5
        )
        # Save legend figure
        legend_fig.savefig(file_name + '_legend.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
        plt.close(legend_fig)
            
    else:
        sns.scatterplot(x="pm1", y="pm2",
                        data=df, ax=ax, s=ms)
    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')  

    labels_list = np.unique(labels)
    #for l in labels_list:
#         #i = np.random.choice(np.where(labels == l)[0])
        #ix_l = np.where(labels == l)[0]
        #c1 = np.median(x[ix_l, 0])
        #c2 = np.median(x[ix_l, 1])
        #ax.text(c1, c2, l, fontsize=fs)


    if file_name:
        plt.savefig(file_name + '.png', format='png', dpi=400)

    plt.close(fig)


def plotPoincareDisc(x,
                     label_names=None,
                     file_name=None,
                     title_name=None,
                     idx_zoom=None,
                     show=False,
                     d1=12,
                     d2=6,
                     fs=6,
                     ms=2,
                     col_palette=None,
                     color_dict=None):
    if col_palette is None:
        col_palette = colors_palette
        # col_palette = plt.get_cmap("tab10")

    df = pd.DataFrame(dict(x=x[0], y=x[1], label=label_names))
    groups = df.groupby(label_names, sort=False) #df.groupby('label')

    fig = plt.figure(figsize=(d1, d2), dpi=300)
    circle = plt.Circle((0, 0), radius=1,  fc='none', color='black')

    plt.subplot(1, 2, 1)
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
    plt.title(title_name, fontsize=fs)

    if color_dict is None:
        j = 0
        color_dict = {}
        for name, group in groups:
            color_dict[name] = col_palette[j]
            j += 1

    marker = 'o'
    for name, group in groups:        
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=color_dict[name], linestyle='', ms=ms, label=name)
    plt.plot(0, 0, 'x', c=(1, 1, 1), ms=ms)
    plt.axis('off')
    plt.axis('equal')
    # plt.legend(numpoints=1, loc='center left',
    #            bbox_to_anchor=(1, 0.5), fontsize=fs)

    labels_list = np.unique(label_names)

    #for l in labels_list:
#         i = np.random.choice(np.where(labels == l)[0])
        #ix_l = np.where(label_names == l)[0]
        #c1 = np.median(x[0, ix_l])
        #c2 = np.median(x[1, ix_l])
        #plt.text(c1, c2, l, fontsize=fs)
#
    if idx_zoom is None:
        xl = np.array(linear_scale(x))
        xl[np.isnan(xl)] = 0

        df = pd.DataFrame(dict(x=xl[0], y=xl[1], label=label_names))
        groups = df.groupby(label_names, sort=False) #df.groupby('label')
    else:
        xl = np.array(linear_scale(x[:, idx_zoom]))
        xl[np.isnan(xl)] = 0

        df = pd.DataFrame(dict(x=xl[0], y=xl[1], label=label_names[idx_zoom]))
        groups = df.groupby(label_names, sort=False) #df.groupby('label')

    circle = plt.Circle((0, 0), radius=1, fc='none',
                        color='black', linestyle=':')
    plt.subplot(1, 2, 2)
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
    plt.title('zoom in', fontsize=fs)

    for name, group in groups:
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=color_dict[name], linestyle='', ms=ms, label=name)

    plt.plot(0, 0, 'x', c=(1, 1, 1), ms=6)

    plt.axis('off')
    plt.axis('equal')

    plt.legend(numpoints=1, loc='center left',
               bbox_to_anchor=(1, 0.5), fontsize=fs)

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name + '.png', format='png')

    if show:
        plt.show()

    plt.close(fig)

    return color_dict

def cayley_transform(points):
    """Transform from Poincare disk to Poincaré upper half-plane using the Cayley transform."""
    # Assert that all input points are inside the disk
    assert np.all(np.sqrt(np.sum(points**2, axis=1)) < 1), "All points must be inside the unit disk."
    # Convert points to complex numbers
    z = points[:, 0] + 1j * points[:, 1]
    # Apply the Cayley transform
    w = (z + 1j) / (1j*z + 1)
    # Convert back to 2D points
    transformed_points = np.column_stack((w.real, w.imag))
    return transformed_points

def plot_halfplane(x, labels=None, labels_name='labels', labels_order=None,
                       file_name=None, coldict=None,
                       d1=19, d2=18.0, fs=8, ms=2, col_palette=plt.get_cmap("tab10"), bbox=(1.3, 0.7),interpolated=0):
    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])
    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    if not (labels is None):
        df[labels_name] = labels[idx]
        if labels_order is None:
            labels_order = np.unique(labels)
        if coldict is None:
            coldict = dict(zip(labels_order, col_palette[:len(labels)]))
        sns.scatterplot(x="pm1", y="pm2", hue=labels_name,
                        hue_order=labels_order,
                        palette=coldict,
                        alpha=0.5, edgecolor="none",
                        data=df, ax=ax, s=ms)

        ax.legend(fontsize=fs, loc='best', bbox_to_anchor=bbox)

    fig.tight_layout()
    #ax.axis('off')
    ax.axis('equal')

    #plt.plot(interpolated[:,0],interpolated[:,1],'k-',linewidth=3,alpha=0.9)
    #plt.plot(interpolated[:,0],interpolated[:,2],'k-',linewidth=3,alpha=0.9)

    # plt.plot(u[0],u[1],'k*',ms=10)
    # plt.plot(v[0],v[1],'k*',ms=10)

    labels_list = np.unique(labels)
    if file_name:
        plt.savefig(f'{file_name}.png', format='png', dpi=400)


def plot_interpolation2(
        x,
        labels=None, labels_name='labels', labels_order=None,
        file_name=None, coldict=None,
        d1=19, d2=18.0, fs=11, ms=5, col_palette=plt.get_cmap("tab10"), bbox=(1.3, 0.7),
        u=None,
        v=None,
        interp_coords=None,
        u_color=None,
        v_color=None
    ):
    idx = np.random.permutation(len(x))
    df = pd.DataFrame(x[idx, :], columns=['pm1', 'pm2'])

    fig = plt.figure(figsize=(d1, d2))
    ax = plt.gca()
    circle = plt.Circle((0, 0), radius=1, fc='none', color='black')
    ax.add_patch(circle)
    ax.plot(0, 0, '.', c=(0, 0, 0), ms=4)


    df[labels_name] = labels[idx]
    if labels_order is None:
        labels_order = np.unique(labels)
    if coldict is None:
        coldict = dict(zip(labels_order, col_palette[:len(labels)]))
    sns.scatterplot(x="pm1", y="pm2", hue=labels_name,
                    hue_order=labels_order,
                    palette=coldict,
                    data=df, ax=ax, s=ms)
    ax.scatter(u[0], u[1], marker='X', s= ms, c=u_color, label='Start')
    ax.scatter(v[0], v[1], marker='X', s= ms, c=v_color, label='End')
    # Plot the interpolated path
    sns.lineplot(x=interp_coords[:, 0], y=interp_coords[:, 1], color='black', linewidth=1, alpha=0.7, label='Interpolation',
                 ax=ax)

    # ax.legend(fontsize=fs, loc='best', bbox_to_anchor=bbox)
    print("plotting legend")
    handles, leg_labels = ax.get_legend_handles_labels()
    if labels_name in leg_labels:
        idx_title = leg_labels.index(labels_name)
        handles.pop(idx_title)
        leg_labels.pop(idx_title)

    # Remove the legend from the main plot
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # ---- Create a separate legend figure ----
    n_items = len(leg_labels)
    ncols = (n_items + 1) // 15  # two rows

    legend_fig = plt.figure(figsize=(ncols * 1.2, 1.6))  # adjust size to fit
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")

    legend = legend_ax.legend(
        handles, leg_labels,
        loc="center",
        ncol=ncols,
        frameon=False,
        fontsize=12,
        markerscale=5
    )
    # Save legend figure
    legend_fig.savefig(file_name + '_legend.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close(legend_fig)

    fig.tight_layout()
    ax.axis('off')
    ax.axis('equal')

    labels_list = np.unique(labels)
    # for l in labels_list:
    #         #i = np.random.choice(np.where(labels == l)[0])
    # ix_l = np.where(labels == l)[0]
    # c1 = np.median(x[ix_l, 0])
    # c2 = np.median(x[ix_l, 1])
    # ax.text(c1, c2, l, fontsize=fs)

    if file_name:
        plt.savefig(file_name + '.png', format='png', dpi=400)

    plt.close(fig)

def plot_interpolation(x,
                     label_names=None,
                     file_name=None,
                     title_name=None,
                     idx_zoom=None,
                     show=False,
                     d1=12,
                     d2=6,
                     fs=6,
                     ms=2,
                     col_palette=None,
                     color_dict=None, u=None, v=None, interp_coords=None,
                       u_color=None, v_color=None):
    if col_palette is None:
        col_palette = colors_palette
        # col_palette = plt.get_cmap("tab10")

    df = pd.DataFrame(dict(x=x[0], y=x[1], label=label_names))
    groups = df.groupby(label_names, sort=False)  # df.groupby('label')

    fig = plt.figure(figsize=(d1, d2), dpi=300)
    circle = plt.Circle((0, 0), radius=1, fc='none', color='black')

    plt.subplot(1, 2, 1)
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
    plt.title(title_name, fontsize=fs)

    if color_dict is None:
        j = 0
        color_dict = {}
        for name, group in groups:
            color_dict[name] = col_palette[j]
            j += 1

    marker = 'o'
    for name, group in groups:
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=color_dict[name], linestyle='', ms=ms, label=name)
    plt.plot(0, 0, 'x', c=(1, 1, 1), ms=ms)
    plt.axis('off')
    plt.axis('equal')
    # plt.legend(numpoints=1, loc='center left',
    #            bbox_to_anchor=(1, 0.5), fontsize=fs)

    # Plot the start and end points of interpolation
    plt.plot(u[0], u[1], color=u_color, marker='X', ms=4 * ms, label='Start')
    plt.plot(v[0], v[1], color=v_color, marker='X', ms=4 * ms, label='End')
    # Plot the interpolated path
    plt.plot(interp_coords[:, 0], interp_coords[:, 1], 'black', linewidth=1, alpha=0.7, label='Interpolation')

    # another legend box


    plt.legend(numpoints=1, loc='center left',
               bbox_to_anchor=(1, 0.5), fontsize=12, markerscale=5)

    labels_list = np.unique(label_names)

    # for l in labels_list:
    #         i = np.random.choice(np.where(labels == l)[0])
    # ix_l = np.where(label_names == l)[0]
    # c1 = np.median(x[0, ix_l])
    # c2 = np.median(x[1, ix_l])
    # plt.text(c1, c2, l, fontsize=fs)
    #
    if idx_zoom is None:
        xl = np.array(linear_scale(x))
        xl[np.isnan(xl)] = 0

        df = pd.DataFrame(dict(x=xl[0], y=xl[1], label=label_names))
        groups = df.groupby(label_names, sort=False)  # df.groupby('label')
    else:
        xl = np.array(linear_scale(x[:, idx_zoom]))
        xl[np.isnan(xl)] = 0

        df = pd.DataFrame(dict(x=xl[0], y=xl[1], label=label_names[idx_zoom]))
        groups = df.groupby(label_names, sort=False)  # df.groupby('label')

    circle = plt.Circle((0, 0), radius=1, fc='none',
                        color='black', linestyle=':')
    plt.subplot(1, 2, 2)
    plt.gca().add_patch(circle)
    plt.plot(0, 0, 'x', c=(0, 0, 0), ms=ms)
    plt.title('zoom in', fontsize=fs)

    for name, group in groups:
        plt.plot(group.x, group.y, marker=marker, markerfacecolor='none',
                 c=color_dict[name], linestyle='', ms=ms, label=name)

    plt.plot(0, 0, 'x', c=(1, 1, 1), ms=6)

    plt.axis('off')
    plt.axis('equal')

    plt.legend(numpoints=1, loc='center left',
               bbox_to_anchor=(1, 0.5), fontsize=12, markerscale=5)

    plt.tight_layout()

    if file_name:
        plt.savefig(file_name + '.png', format='png')

    if show:
        plt.show()

    plt.close(fig)

    return color_dict