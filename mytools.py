# mytools.py

import umap
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from urllib import request
from urllib.request import urlopen
from rdkit import Chem

from torch_geometric.data import Data, Dataset
import torch
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)
from tqdm import tqdm


"""画图方法"""

def v_umap(data, labels=None):
    """
    对输入的数据data应用UMAP降维，并可视化结果。
    
    参数:
    - data: 要降维的数据集，形状应为(n_samples, n_features)。
    - labels: 每个样本对应的标签（可选），形状为(n_samples,)，用于以不同颜色显示不同的类别。
    
    返回:
    - umap_plot: UMAP降维后数据的二维散点图。
    """
    # 应用UMAP算法
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42, n_jobs=1)
    data_2d = reducer.fit_transform(data)

    # 创建图形
    plt.figure(figsize=(10, 7))
    if labels is not None:
        sns.scatterplot(x=data_2d[:,0], y=data_2d[:,1], hue=labels, palette='bright')
    else:
        plt.scatter(data_2d[:,0], data_2d[:,1], alpha=0.7, s=10)
    
    plt.title('UMAP Visualization')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # 显示图形
    return plt.show()
##---------------------------------------------------------------------------------------------------------------##

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    绘制95%置信椭圆
    """
    ax = ax or plt.gca()
    
    # 计算协方差矩阵的特征值和特征向量
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # 计算椭圆的角度和尺寸
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals) * np.sqrt(chi2.ppf(0.95, df=2))

    # 创建椭圆对象
    ellipse = Ellipse(xy=position, width=width, height=height, angle=theta, facecolor='none', **kwargs)

    # 添加椭圆到图表
    ax.add_artist(ellipse)
    return ellipse

def v_pca(data):
    """
    使用PCA对输入数据进行降维，并在2D图中可视化结果
    """
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(standardized_data)
    
    df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    cov_matrix = np.cov(df_pca['PC1'], df_pca['PC2'])

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(x=df_pca['PC1'], y=df_pca['PC2'], s=10, alpha=0.6, label='Samples')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')

    draw_ellipse((0, 0), cov_matrix, ax=ax, edgecolor='red', alpha=0.8, label='95% Confidence Ellipse')

    ax.set_title('PCA 2D Visualization with 95% Confidence Ellipse')

    x_min, x_max = df_pca['PC1'].min(), df_pca['PC1'].max()
    y_min, y_max = df_pca['PC2'].min(), df_pca['PC2'].max()

    scale_x = np.sqrt(cov_matrix[0, 0]) * chi2.ppf(0.95, df=2)
    scale_y = np.sqrt(cov_matrix[1, 1]) * chi2.ppf(0.95, df=2)

    x_margin = max(abs(x_min), abs(x_max), scale_x) * 1.1
    y_margin = max(abs(y_min), abs(y_max), scale_y) * 1.1

    ax.set_xlim([-x_margin, x_margin])
    ax.set_ylim([-y_margin, y_margin])

    plt.legend()
    
    return plt.show()

##---------------------------------------------------------------------------------------------------------------##
"""常用工具"""

def get_smiles(substrate):
    '''query SMILES string of the substrate from PubChem
    :param substrate: 'string' substrate name
    :return 'string' SMILES string
    '''
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT'%substrate
        req = requests.get(url)
        if req.status_code != 200:
            smiles = 'NaN'
        else:
            smiles = req.content.splitlines()[0].decode()
    except :
        smiles = 'NaN'
    return smiles

def get_p_seq(protID):
    '''Query protein sequence from uniprot.
    :param ID: 'string' uniprot id of protein
    :return 'string' protein sequence
    '''
    url = "https://www.uniprot.org/uniprot/%s.fasta" % ID
    try :
        data = requests.get(url)
        if data.status_code != 200:
            seq = 'NaN'
        else:
            seq =  "".join(data.text.split("\n")[1:])
    except :
        seq = 'NaN'
    return seq

def get_p_mw(protID):
    '''query protein molar weight from uniprot
    :param protID: 'string' uniprot id of protein
    :return 'int' molar weight
    '''
    data = urlopen("http://www.uniprot.org/uniprot/" + protID + ".txt").read().decode()
    result = data.split('SQ   ')[-1]
    mw = int(result.split(';')[1].strip().split()[0])
    return mw

def create_m_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency

def scale_minmax(array):
    '''
    Normalize the feature as x-x_min/x_max-x_min.
    '''
    x_min = np.min(array)
    x_max = np.max(array)
    scaled_array = [(x-x_min)/(x_max-x_min) for x in array]
    return scaled_array

def check_mutations(seq, mut_list):
    no_error = True
    for mut in mut_list:
        ind = int(mut[1:-1])-1
        old = mut[0].upper()
        if (ind > len(seq)-1) or (seq[ind] != old):
            no_error = False
            break
    return no_error

def apply_mutations(seq, mut_list):
    mut_seq = seq
    for mut in mut_list:
        ind = int(mut[1:-1])-1
        new = mut[-1].upper()
        temp_list = list(mut_seq)
        temp_list[ind] = new
        mut_seq = ''.join(temp_list)
    return mut_seq


def ReorderCanonicalRankAtoms(mol):
    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))])))[1]
    mol_renum = Chem.RenumberAtoms(mol, order)
    return mol_renum, order

def get_a_features(smile, removeHs=True, reorder_atoms=False):
    # atoms

    mol = Chem.MolFromSmiles(smile)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    return x

def smiles2graph(smiles_string, removeHs=True, reorder_atoms=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = mol if removeHs else Chem.AddHs(mol)
    if reorder_atoms:
        mol, _ = ReorderCanonicalRankAtoms(mol)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    return Data(x=torch.tensor(x, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32))

def split_data( df, ratio=0.1):
    '''
    Randomly split data into two datasets.
    '''
    idx = np.arange(df.shape[0])
    np.random.shuffle(idx)
    num_split = int(df.shape[0] * ratio)
    idx_test, idx_train = idx[:num_split], idx[num_split:]
    df_test = df.iloc[idx_test].reset_index().drop(['index'],axis=1)
    df_train = df.iloc[idx_train].reset_index().drop(['index'],axis=1)
    return df_test, df_train

##---------------------------------------------------------------------------------------------------------------##

""" 序列嵌入 """
## 构建词典
def build_word_dict(sequences, ngram):
    """从序列集合中提取所有可能的ngram子序列，并为每个唯一的子序列分配一个唯一的编号"""
    
    # 使用集合去重并确保序列唯一
    sequences = list(set(sequences))
    
    word_dict = {}
    next_id = 0
    
    for sequence in tqdm(sequences, desc='处理中:'):
        # 在序列前后分别添加特殊字符 '>' 和 '<'
        extended_sequence = '>' + sequence + '<'
        
        # 提取所有长度为ngram的子序列
        ngrams = [extended_sequence[i:i+ngram] for i in range(len(extended_sequence) - ngram + 1)]
        
        for ngram_item in ngrams:
            if ngram_item not in word_dict:
                word_dict[ngram_item] = next_id
                next_id += 1
    
    return word_dict

## 序列到索引
def check_dict( item, dict2check ):
    if item in dict2check.keys():
        return dict2check[item]
    else:
        if len(dict2check.keys()) == 0:
            dict2check[item] = 0
        else:
            dict2check[item] = max(list(dict2check.values())) + 1
    return dict2check[item]


def split_sequence(sequence, ngram, word_dict):
    sequence = '>' + sequence + '<'  # 添加起始和结束标记
    words = [check_dict(sequence[i:i+ngram], word_dict) for i in range(len(sequence)-ngram+1)]
    return np.array(words)

""" smiles 嵌入 """
def get_sent_feat(sentence, model):
    vectors = []
    for word in sentence:
        try:
            vector = model.wv[word]
            vectors.append(vector)
        except KeyError:
            # 如果词不在词汇表中，则跳过
            continue
    if not vectors:
        return None  # 如果没有有效的词向量，返回 None
    # 将列表转换为 NumPy 数组并调整形状
    vectors_np = torch.tensor(np.array(vectors), dtype=torch.float32)
    return vectors_np

