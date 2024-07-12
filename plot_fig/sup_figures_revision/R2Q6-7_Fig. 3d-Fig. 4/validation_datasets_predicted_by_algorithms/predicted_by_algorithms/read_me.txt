SC_ovarian_canc，SC_glioblastoma 数据集的 kassandra 的预测结果， tumor 重新设置的，设置的和other 一样的值 



原始数据：
SC_ovarian_canc 数据集 的细胞类型删掉 “Erythrocytes”（红细胞）。 留下：B_cells, Dendritic_cells, Fibroblasts, Macrophages, NK_cells, T_cells, Tumor
SC_glioblastoma数据集的细胞类型删掉 “Oligodendrocytes”。 留下：Macrophages, Tumor, T_cells
SC_HNSCC 数据集的细胞类型删掉 Tregs，T_helpers。 留下：CD4_T_cells, Dendritic_cells, Endothelium, Fibroblasts, Macrophages, Mast_cells, Myocytes, Myofibroblasts, Non_plasma_B_cells, Plasma_B_cells, Tumor, CD8_T_cells,T_cells, B_cells,Other,Lymphocytes
GSE121127 数据集的细胞类型：Tumor, Fibroblasts, Lymphocytes. 注：kassandra 预测的这个数据集没有Other 的数据




'Plasma_B_cells':'Plasma B cells','Non_plasma_B_cells':'Non-plasma B cells','B_cells':'B Cells','CD4_T_cells':'CD4 T','CD8_T_cells':'CD8 T','T_cells':'T Cells','Dendritic_cells':'DC','NK_cells':'NK','Neutrophils':'Neutrophils','Monocytes','Lymphocytes':'Lymphocytes','Other':'Cancer Cells','Macrophages':'Macrophages', 'Mast_cells':'Mast Cells', 'Tumor':'Cancer Cells','Fibroblasts':'Fibroblasts','CAFs':'CAFs', 'Myofibroblasts':'Myofibroblasts','Endothelium':'Endothelial Cells'
------------------------

'Plasma_B_cells','Non_plasma_B_cells','B_cells','CD4_T_cells','CD8_T_cells','T_cells','Dendritic_cells','NK_cells','Neutrophils','Monocytes','Lymphocytes','Other'

'Plasma_B_cells','T_cells', 'Macrophages', 'NK_cells',
                           'Dendritic_cells', 'Mast_cells', 'B_cells', 'CD4_T_cells','CD8_T_cells',
                           'Lymphocytes', 'Non_plasma_B_cells','Monocytes','Neutrophils'

'Plasma B cells', 'Non-plasma B cells', 'CD4 T',  'DC', 'Endothelial Cells',
                            'Macrophages', 'Mast Cells', 'NK','Neutrophils', 'Monocytes',  'Cancer Cells',   #'1-others',
                                              'CD8_T_cells','B_cells','T_cells','Fibroblasts','CAFs', 'Myofibroblasts','Lymphocytes'


'B_cells', 'CD4_T_cells', 'CD8_T_cells', 'Monocytes', 'Neutrophils',
                           'NK_cells', 'Other', 'T_cells', 'Lymphocytes'



'B_cells', 'Fibroblasts', 'CD4_T_cells', 'CD8_T_cells', 'Endothelium',
                           'Macrophages', 'NK_cells', 'Other', 'T_cells', 'Lymphocytes'



'Neutrophils','T_cells', 'CD4_T_cells','CD8_T_cells', 'Macrophages', 'NK_cells', 'B_cells',
                           'Dendritic_cells', 'Lymphocytes', 'Non_plasma_B_cells','Plasma_B_cells','Monocytes','Other'


'Plasma_B_cells', 'CD8_T_cells','Neutrophils',
                           'T_cells', 'CD4_T_cells', 'Macrophages', 'NK_cells', 'B_cells',
                           'Dendritic_cells', 'Lymphocytes', 'Non_plasma_B_cells','Monocytes'



'CD8_T_cells', 'CD4_T_cells', 'Fibroblasts', 'Macrophages', 'B_cells',
                           'Tumor', 'Mast_cells', 'Dendritic_cells', 'Endothelium',
                           'T_cells'


'B_cells',  'CD4_T_cells',
                               'CD8_T_cells', 'Endothelium',  'Fibroblasts', 
                               'Lymphocytes',  'Macrophages', 'Monocytes',  'NK_cells',
                                'Neutrophils',  'Non_plasma_B_cells','Plasma_B_cells',
                                'T_cells','Other'


'Plasma_B_cells', 'CD8_T_cells',
                           'Monocytes', 'Neutrophils',  'B_cells',
                           'CD4_T_cells', 'T_cells', 'NK_cells', 'Macrophages', 'Dendritic_cells',
                           'Mast_cells', 'Non_plasma_B_cells', 'Lymphocytes', 'Other'



'B_cells', 'Monocytes',
                           'Neutrophils', 'NK_cells', 'CD8_T_cells',
                           'Dendritic_cells', 'T_cells', 'Macrophages', 'CD4_T_cells',
                           'Lymphocytes','Other'

'B_cells',  'Monocytes',
                           'Neutrophils', 'NK_cells', 'CD8_T_cells', 
                           'Dendritic_cells', 'T_cells', 'Macrophages', 'CD4_T_cells',
                           'Lymphocytes', 'Other'

'Monocytes',  'CD4_T_cells', 'B_cells', 'NK_cells',
                           'CD8_T_cells',







