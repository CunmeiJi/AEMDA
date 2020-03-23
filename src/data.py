import pickle
import xlrd
import numpy as np
import torch
import pickle
import os

dir_path = '../data/'

#1. load all disease name,
workbook = xlrd.open_workbook(dir_path + 'disease name.xlsx')
sheet = workbook.sheets()[0] #
tmplist = sheet.col_values(1)
disease_name = [str(dis).lower() for dis in tmplist]

#2. load all mirna name,
workbook = xlrd.open_workbook(dir_path + 'miRNA name.xlsx')
sheet = workbook.sheets()[0] #
tmplist = sheet.col_values(1)
mirna_name = [str(mi).lower() for mi in tmplist]

#3. load disease semantic similarity.
dis_sim = np.loadtxt(dir_path + 'disease_semantic_similarity.txt',dtype=np.float32)
dis_sim = torch.from_numpy(dis_sim)

#4. load miRNA functional similarity 
mi_sim = np.loadtxt(dir_path + 'miRNA_functional_similarity.txt', dtype=np.float32)
mi_sim = torch.from_numpy(mi_sim)

#5. load gaussian disease simlarity 
dis_sim_gaussian = np.loadtxt(dir_path + 'gaussian_disease.txt', dtype=np.float32)
dis_sim_gaussian = torch.from_numpy(dis_sim_gaussian)

#6. load gaussian miRNA simlarity 
mi_sim_gaussian = np.loadtxt(dir_path + 'gaussian_miRNA.txt', dtype=np.float32)
mi_sim_gaussian = torch.from_numpy(mi_sim_gaussian)

#5. load known dis-miRNA interaction
dis_mi = np.loadtxt(dir_path + 'known_disease_miRNA_interaction.txt',dtype=np.int32)
dis_mi = torch.from_numpy(dis_mi)
dis_mi = dis_mi.long()
