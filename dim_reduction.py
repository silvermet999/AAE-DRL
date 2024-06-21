import main
from sklearn.decomposition import PCA

# disciminant analysis

def PCA_alg(df):
    pca = PCA(n_components=100)
    df = pca.fit_transform(df)
    return df

x_pca_train = PCA_alg(main.x_train)
x_pca_test = PCA_alg(main.x_test)