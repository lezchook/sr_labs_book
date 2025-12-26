# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    # Here is your code
    for score, label in zip(all_scores, all_labels):
        if label == 1:
            tar_scores.append(score)
        else:
            imp_scores.append(score)
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)  # Среднее таргет-оценок
    tar_scores_std  = np.std(tar_scores)   # Стандартное отклонение таргет-оценок
    imp_scores_mean = np.mean(imp_scores)  # Среднее импостор-оценок
    imp_scores_std  = np.std(imp_scores)   # Стандартное отклонение импостор-оценок
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    # Сортировка оценок по возрастанию
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    sort_idx = np.argsort(all_scores)
    all_scores_sort = all_scores[sort_idx]
    ground_truth_sort = all_labels[sort_idx].astype(bool)
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code

    # Вычисление гауссовских PDF для каждой оценки
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)

    # Вычисление логарифма отношения правдоподобия (LLR)
    LLR = np.log(tar_gauss_pdf / imp_gauss_pdf)
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = 0.0
    fpr   = 0.0
    
    ###########################################################
    # Here is your code
    # Сортировка LLR и меток по возрастанию LLR
    sort_idx = np.argsort(LLR)
    LLR_sort = LLR[sort_idx]
    gt_sort  = ground_truth_sort[sort_idx] # Соответствующие метки

    n_tar = np.sum(gt_sort)               # Общее число таргет-пар
    n_false = int(np.floor(fnr * n_tar))  # Число допустимых ложных отрицательных (FN)

    # Выбор порога LLR для заданной FNR
    tar_indices = np.where(gt_sort)[0]  # Индексы таргет-пар в отсортированном массиве
    if n_false == 0:
        # Если не допускается ни одной ошибки первого рода, берём порог чуть ниже минимального LLR таргет-пары
        thr = LLR_sort[tar_indices[0]]
    else:
        # Порог выбираем так, чтобы ровно n_false таргет-пар оказались ниже порога
        thr = LLR_sort[tar_indices[n_false-1]]
    
    # Вычисление вероятности ошибки второго рода (FPR)
    imp_indices = np.where(~gt_sort)[0]  # индексы импосторов
    
    # Считаем, сколько импостор-пар выше порога (ошибка второго рода)
    fpr = np.sum(LLR_sort[imp_indices] > thr) / len(imp_indices)
    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    # Вычисление априорной вероятности импостор-пары
    P_Himp = 1.0 - P_Htar
    
    # Вычисление порога по критерию Байеса
    # thr = ln((C01 - C11) * P(H1) / [(C10 - C00) * P(H0)])
    thr = np.log((C01 - C11) * P_Himp / ((C10 - C00) * P_Htar))
    
    # Принятие решений на основе порога
    # Если LLR > thr, то решение в пользу таргет-пары (D0)
    # Если LLR <= thr, то решение в пользу импостор-пары (D1)
    decisions = (LLR > thr).astype(int)  # 1 - таргет (D0), 0 - импостор (D1)
    
    # Подсчет ошибок
    # Ошибка первого рода (FRR): таргет-пара распознана как импостор
    # ground_truth_sort == 1 (таргет), но decisions == 0 (решение импостор)
    fnr_count = np.sum((ground_truth_sort == 1) & (decisions == 0))
    num_tar = np.sum(ground_truth_sort == 1)
    fnr = fnr_count / num_tar if num_tar > 0 else 0.0
    
    # Ошибка второго рода (FAR): импостор-пара распознана как таргет
    # ground_truth_sort == 0 (импостор), но decisions == 1 (решение таргет)
    fpr_count = np.sum((ground_truth_sort == 0) & (decisions == 1))
    num_imp = np.sum(ground_truth_sort == 0)
    fpr = fpr_count / num_imp if num_imp > 0 else 0.0
    
    # Вычисление средней стоимости (байесовского риска)
    # Условные вероятности:
    P_D0_given_H0 = 1.0 - fnr  # правильное положительное решение
    P_D1_given_H0 = fnr        # ошибка первого рода
    P_D0_given_H1 = fpr        # ошибка второго рода
    P_D1_given_H1 = 1.0 - fpr  # правильное отрицательное решение
    
    AC = (C00 * P_D0_given_H0 * P_Htar + 
          C10 * P_D1_given_H0 * P_Htar + 
          C01 * P_D0_given_H1 * P_Himp + 
          C11 * P_D1_given_H1 * P_Himp)
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = 0.0
    P_Htar = 0.0
    
    ###########################################################
    # Here is your code
    P_values = np.linspace(0.01, 0.99, 99)
    max_risk = -np.inf
    best_P = 0.5
    best_thr = 0.0
    
    for P in P_values:
        P_imp = 1.0 - P
        
        # Вычисление порога по критерию Байеса для текущего P
        thr_tmp = np.log((C01 - C11) * P_imp / ((C10 - C00) * P))
        
        # Принятие решений на основе текущего порога
        decisions = (LLR > thr_tmp).astype(int)
        
        # Подсчет ошибок для текущего порога
        fnr_tmp_count = np.sum((ground_truth_sort == 1) & (decisions == 0))
        num_tar = np.sum(ground_truth_sort == 1)
        fnr_tmp = fnr_tmp_count / num_tar if num_tar > 0 else 0.0
        
        fpr_tmp_count = np.sum((ground_truth_sort == 0) & (decisions == 1))
        num_imp = np.sum(ground_truth_sort == 0)
        fpr_tmp = fpr_tmp_count / num_imp if num_imp > 0 else 0.0
        
        # Вычисление байесовского риска для текущего P
        P_D0_given_H0 = 1.0 - fnr_tmp
        P_D1_given_H0 = fnr_tmp
        P_D0_given_H1 = fpr_tmp
        P_D1_given_H1 = 1.0 - fpr_tmp
        
        risk = (C00 * P_D0_given_H0 * P + 
                C10 * P_D1_given_H0 * P + 
                C01 * P_D0_given_H1 * P_imp + 
                C11 * P_D1_given_H1 * P_imp)
        
        # Ищем максимум риска
        if risk > max_risk:
            max_risk = risk
            best_P = P
            best_thr = thr_tmp
    
    # Используем найденное оптимальное значение P_Htar и порог
    P_Htar = best_P
    P_Himp = 1.0 - P_Htar
    thr = best_thr
    
    # Вычисление финальных значений FNR и FAR для найденного порога
    decisions = (LLR > thr).astype(int)
    
    fnr_count = np.sum((ground_truth_sort == 1) & (decisions == 0))
    num_tar = np.sum(ground_truth_sort == 1)
    fnr = fnr_count / num_tar if num_tar > 0 else 0.0
    
    fpr_count = np.sum((ground_truth_sort == 0) & (decisions == 1))
    num_imp = np.sum(ground_truth_sort == 0)
    fpr = fpr_count / num_imp if num_imp > 0 else 0.0
    
    # Вычисление финальной средней стоимости
    P_D0_given_H0 = 1.0 - fnr
    P_D1_given_H0 = fnr
    P_D0_given_H1 = fpr
    P_D1_given_H1 = 1.0 - fpr
    
    AC = (C00 * P_D0_given_H0 * P_Htar + 
          C10 * P_D1_given_H0 * P_Htar + 
          C01 * P_D0_given_H1 * P_Himp + 
          C11 * P_D1_given_H1 * P_Himp)
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar