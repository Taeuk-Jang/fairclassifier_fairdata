import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, dataset):
        self.label = dataset.labels.squeeze(-1).astype(int)
        self.feature = dataset.features
        #n_values = int(np.max(self.label) + 1)
        #self.label = np.eye(n_values)[self.label.astype(int)].squeeze(1)
        
    def __getitem__(self, idx):
    
        y = self.label[idx]
        x = self.feature[idx]
        
        return x, y
    
    
    def __len__(self):
        return len(self.label)
    

def fpr_calc(label, pred):
    FP = (sum(pred[(label == 0)])) #False Positive
    TN = (len(pred[(label == 0)]) - sum(pred[(label == 0)]))

    fpr = FP / (FP+TN)
    
    return fpr

def tpr_calc(label, pred):
    TP = (sum(pred[(label == 1)])) #False Positive
    FN = (len(pred[(label == 1)]) - sum(pred[(label == 1)]))

    tpr = TP / (TP+FN)
    
    return tpr

def evaluate(classifier, testloader):
    tpr_overall, tpr_priv, tpr_unpriv,\
    fpr_overall, fpr_unpriv, fpr_priv,\
    acc_overall, acc_priv, acc_unpriv, eq_overall = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    cnt = 0

    tp_priv, tn_priv, fp_priv, fn_priv, \
    tp_unpriv, tn_unpriv, fp_unpriv, fn_unpriv = 0, 0, 0, 0, 0, 0, 0, 0

    for x, a, y in testloader:
        x, a, y = x.to(device).float(), a.view(-1,1)[:,0].to(device).float(), y.view(-1,1).to(device).float()

        classifier.eval()
        
        pred_test = classifier(x)
        pred_test = F.sigmoid(pred_test)
        
        loss_test = criterion(pred_test, y)

        pred_test[(pred_test)>0.5] = 1
        pred_test[(pred_test)<=0.5] = 0

        priv_idx = (a==1).squeeze()
        positive_idx = y==1

        test_lb_priv = y[priv_idx]
        test_lb_unpriv = y[~priv_idx]

        pred_priv = pred_test[priv_idx]
        pred_unpriv = pred_test[~priv_idx]

        tp_priv += sum(pred_priv[test_lb_priv == 1] == 1)
        fp_priv += sum(pred_priv[test_lb_priv == 0] == 1)
        tn_priv += sum(pred_priv[test_lb_priv == 0] == 0)
        fn_priv += sum(pred_priv[test_lb_priv == 1] == 0)

        tp_unpriv += sum(pred_unpriv[test_lb_unpriv == 1] == 1)
        fp_unpriv += sum(pred_unpriv[test_lb_unpriv == 0] == 1)
        tn_unpriv += sum(pred_unpriv[test_lb_unpriv == 0] == 0)
        fn_unpriv += sum(pred_unpriv[test_lb_unpriv == 1] == 0)

    tpr_overall = (tp_priv + tp_unpriv)/(tp_priv + tp_unpriv + fn_priv + fn_unpriv).float().item()
    tpr_unpriv = (tp_unpriv)/(tp_unpriv + fn_unpriv).float().item()
    tpr_priv = (tp_priv)/(tp_priv + fn_priv).float().item()

    fpr_overall = (fp_priv + fp_unpriv)/(tn_priv + tn_unpriv + fp_priv + fp_unpriv).float().item()
    fpr_unpriv = (fp_unpriv)/(tn_unpriv + fp_unpriv).float().item()
    fpr_priv = (fp_priv)/(tn_priv + fp_priv).float().item()

    acc_overall = (tp_priv + tn_priv + tp_unpriv + tn_unpriv)/(tp_priv + tn_priv + tp_unpriv + tn_unpriv + \
                                                              fp_priv + fn_priv + fp_unpriv + fn_unpriv).float().item()
    acc_priv = (tp_priv + tn_priv)/(tp_priv + tn_priv + fp_priv + fn_priv).float().item()
    acc_unpriv = (tp_unpriv + tn_unpriv)/(tp_unpriv + tn_unpriv + fp_unpriv + fn_unpriv).float().item()

        #eq_overall = abs()

    #     print('\n epoch {}-{}'.format(repeat, epoch))
    print()
    print('overall TPR : {0:.3f}'.format( tpr_overall))
    print('priv TPR : {0:.3f}'.format( tpr_priv))
    print('unpriv TPR : {0:.3f}'.format( tpr_unpriv))
    print('Eq. Opp : {0:.3f}'.format( abs(tpr_unpriv - tpr_priv)))
    print()
    print('overall FPR : {0:.3f}'.format( fpr_overall))
    print('priv FPR : {0:.3f}'.format( fpr_priv))
    print('unpriv FPR : {0:.3f}'.format( fpr_unpriv))
    print('diff FPR : {0:.3f}'.format( abs(fpr_unpriv-fpr_priv)))
    print()
    print('overall ACC : {0:.3f}'.format( acc_overall))
    print('priv ACC : {0:.3f}'.format( acc_priv))
    print('unpriv ACC : {0:.3f}'.format( acc_unpriv)) 
    print('diff ACC : {0:.3f}\n\n\n'.format( abs(acc_unpriv-acc_priv)))

    test_pred = data_test.copy(deepcopy=True)
    feature_size = test_pred.features.shape[1]
    sens_loc = np.zeros(feature_size).astype(bool)
    sens_loc[sens_idx] = 1

    feature = test_pred.features[:,~sens_loc] #data without sensitive
    feature = min_max_scaler.fit_transform(feature)

    test_pred.labels = F.sigmoid(classifier(torch.tensor(feature).float().to(device))).detach().cpu().numpy()

    test_pred.labels[test_pred.labels>0.5] = 1
    test_pred.labels[test_pred.labels<=0.5] = 0

    classified_metric = ClassificationMetric(data_test,
                                                     test_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

    bal_acc_priv = 1/2*((1-fpr_priv) + tpr_priv)
    bal_acc_unpriv = 1/2*((1-fpr_unpriv) + tpr_unpriv)

    print('balanced acc :' ,1/2*((1-fpr_overall) + tpr_overall))

    print('bal acc diff : {:.3f}'.format(abs(bal_acc_priv - bal_acc_unpriv)))
    print('average_abs_odds_difference : {:.3f}'.format(0.5 * (abs(tpr_priv - tpr_unpriv) + abs(fpr_priv - fpr_unpriv))))
    print('disparate_impact :' ,classified_metric.disparate_impact())
    print('theil_index :' ,classified_metric.theil_index())
    print('statistical_parity_difference :' ,classified_metric.statistical_parity_difference())
    
    return     tpr_overall.item(), tpr_priv.item(), tpr_unpriv.item(),\
    fpr_overall.item(), fpr_unpriv.item(), fpr_priv.item(),\
    acc_overall.item(), acc_priv.item(), acc_unpriv.item(), eq_overall,\
    classified_metric.disparate_impact(), classified_metric.theil_index(), classified_metric.statistical_parity_difference()

