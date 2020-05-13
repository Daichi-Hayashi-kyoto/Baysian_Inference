import numpy as np
import scipy.stats as stats
from scipy.special import digamma
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

def to_one_hot(s, n_clusters):
    return np.identity(n_clusters)[s]

def log_sum_exp(X):
    max_x = np.max(X, axis = 1).reshape(-1, 1)
    return np.log(np.sum(np.exp(X - max_x), axis = 1).reshape(-1, 1)) + max_x


def variational_inference_Poisson_Mixture(data, K, max_iter):
    "初期化"
    X = data.copy()
    a_init = np.ones([K, 1])
    b_init = np.ones([K, 1])
    alpha_init = np.random.rand(K, 1) # 0以上１未満の乱数を発生
    X = X.reshape(-1, 1)
    a_hat = a_init.copy()
    b_hat = b_init.copy()
    alpha_hat = alpha_init.copy()

    
    for _ in range(max_iter):

        ln_lambda_expectation = digamma(a_hat) - np.log(b_hat)
        lam_expectation = a_hat / b_hat
        ln_pi_expectation = digamma(alpha_hat) - digamma(np.sum(alpha_hat)) # shape = (K, 1)
        # q(S_n)を決定するパラメータetaの更新
        tmp = np.dot(X, ln_lambda_expectation.T) - lam_expectation.reshape(1, -1) + ln_pi_expectation.reshape(1, -1)
        eta = np.exp(tmp - log_sum_exp(tmp))

        #q(λ_k)の更新
        a_hat = np.dot(X.T, eta).reshape(K, 1) + a_init
        b_hat = np.sum(eta, axis = 0, keepdims = True).T + b_init   # 行方向を消すようにsummationをとる. keepdims = Trueでsumしない方向のdimを残している.

        #Dirichlet分布の更新
        alpha_hat = np.sum(eta, axis = 0, keepdims = True).T + alpha_init



    return a_hat, b_hat, alpha_hat, eta



if __name__ == "__main__":
    '''
    サンプルデータの作成
    '''

    random.seed(0)

    n_clusters = 2
    each_cluster_samples = [200, 300]
    n_samples = np.sum(each_cluster_samples)
    pi_true = each_cluster_samples / n_samples

    s_true = to_one_hot(np.concatenate([np.repeat(i, n) for i, n in enumerate(each_cluster_samples)]), n_clusters)
    lam_true = np.array([10, 30])
    x = np.stack([stats.poisson.rvs(lam_k, size=n_samples) for lam_k in lam_true], axis=1)
    x = np.sum(x * s_true, axis=1)

    fig, axes = plt.subplots(1, 2, figsize = (10, 7))
    axes[0].set_title("Sample data", fontsize = 18)
    axes[0].hist(x, bins=int(np.max(x) - np.min(x)), ec = "black")

    _, _, _, eta = variational_inference_Poisson_Mixture(data = x, K = 2, max_iter = 200)

    '''
    Plot
    '''

    # Plot
    x_max = x.max()
    x_min = x.min()
    bins = np.linspace(x_min, x_max, x_max - x_min + 1)
    x_dig = np.digitize(x, bins)

    weight = np.zeros(len(bins))
    count = np.zeros(len(bins))
    for x_dig_i, eta_0_i in zip(x_dig, eta[:, 0]):   # 組み合わせをzipで取ってくる
       weight[x_dig_i - 1] += eta_0_i
       count[x_dig_i - 1] += 1

    weight = [w / c if c != 0 else 0 for w, c in zip(weight, count)]

    _, _, patches = axes[1].hist(x, bins=bins, ec = "black")
    axes[1].set_title("Result in Clustering")
    cm = plt.cm.get_cmap('bwr')
    colors = [cm(w) for w in weight]
    for patch, color in zip(patches, colors):
        patch.set_fc(color)
    
    plt.show()  