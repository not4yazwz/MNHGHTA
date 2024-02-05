# 作者: not4ya
# 时间: 2023/11/15 9:48
import time
from torch import optim
from model import Model
from param import parameter_parser
from util import *

if __name__ == '__main__':
    args = parameter_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    herb_sim, target_sim, samples, pos_edges, neg_edges = load_data()
    metric_list = []
    metrics = np.zeros(7)
    x_ROC_list = []
    x_PR_list = []
    y_ROC_list = []
    y_PR_list = []
    for j in range(1, 6):
        fold_dir = f'fold_{j}'
        pos_edges_train_df = pd.read_csv(f'{fold_dir}/pos_edges_train.csv')
        pos_edges_train = pos_edges_train_df[['herb', 'target']].to_numpy().T
        neg_edges_train_df = pd.read_csv(f'{fold_dir}/neg_edges_train.csv')
        neg_edges_train = neg_edges_train_df[['herb', 'target']].to_numpy().T
        pos_edges_test_df = pd.read_csv(f'{fold_dir}/pos_edges_test.csv')
        pos_edges_test = pos_edges_test_df[['herb', 'target']].to_numpy().T
        neg_edges_test_df = pd.read_csv(f'{fold_dir}/neg_edges_test.csv')
        neg_edges_test = neg_edges_test_df[['herb', 'target']].to_numpy().T

        herb_sim = pd.read_csv(f'{fold_dir}/herb_sim.csv', index_col=False, dtype=np.float32).to_numpy()
        target_sim = pd.read_csv(f'{fold_dir}/target_sim.csv', index_col=False, dtype=np.float32).to_numpy()

        '''Erase known relationships(test)'''
        n_herb = herb_sim.shape[0]
        n_target = target_sim.shape[0]
        new_association = np.zeros((n_herb, n_target))
        new_association[pos_edges_train[0], pos_edges_train[1]] = 1

        '''K-nearest neighbor generation network'''
        herb_adj = k_matrix(herb_sim, args.knn_nums)
        target_adj = k_matrix(target_sim, args.knn_nums)

        '''Initial network and feature construction'''
        het_net = construct_adj_mat(new_association)
        het_net_device = torch.tensor(np.array(np.where(het_net == 1)), dtype=torch.long, device=device)
        het_x = construct_het_mat(new_association, herb_sim, target_sim)
        het_x_device = torch.tensor(het_x, dtype=torch.float32, device=device)

        herb_net = np.array(tuple(np.where(herb_adj != 0)))
        herb_net_device = torch.tensor(herb_net, dtype=torch.long, device=device)
        herb_x_device = torch.tensor(herb_sim, dtype=torch.float32, device=device)

        target_net = np.array(tuple(np.where(target_adj != 0)))
        target_net_device = torch.tensor(target_net, dtype=torch.long, device=device)
        target_x_device = torch.tensor(target_sim, dtype=torch.float32, device=device)

        '''Model Definition'''
        model = Model(args).cuda()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=1e-3, step_size_up=200,
                                                step_size_down=200, mode='exp_range', gamma=0.99, scale_fn=None,
                                                cycle_momentum=False, last_epoch=-1)

        '''Model Training'''
        t_total = time.time()
        model.train()
        max_time = 0
        max_auc = 0
        for epoch in range(args.epoch):
            model.zero_grad()
            predict = model(het_net_device, het_x_device, herb_net_device, herb_x_device,
                            target_net_device, target_x_device)
            predict = predict.cpu().reshape(n_herb, n_target)
            loss = calculate_loss(predict, pos_edges_train, neg_edges_train)
            loss.backward()
            optimizer.step()
            scheduler.step()

        '''Model Test'''
        model.eval()
        with torch.no_grad():
            predict = model(het_net_device, het_x_device, herb_net_device, herb_x_device,
                            target_net_device, target_x_device)
            predict = predict.cpu().detach().reshape(n_herb, n_target)
            metric, x_ROC, y_ROC, x_PR, y_PR = calculate_evaluation_metrics(predict, pos_edges_test, neg_edges_test)
            x_ROC_list.append(x_ROC)
            x_PR_list.append(x_PR)
            y_ROC_list.append(y_ROC)
            y_PR_list.append(y_PR)
            metric_list.append(metric)
            metrics = [m + n for m, n in zip(metric, metrics)]
            print('fold_{}-auc:{:.4f},aupr:{:.4f},f1_score:{:.4f},accuracy:{:.4f},recall:{:.4f},specificity:{:.4f},precision:{:.4f}'.format(j, metric[0],
                                                                                                                                    metric[1],
                                                                                                                                    metric[2],
                                                                                                                                    metric[3],
                                                                                                                                    metric[4],
                                                                                                                                    metric[5],
                                                                                                                                    metric[6]))
    metrics = [value / 5 for value in metrics]
    print('auc:{:.4f},aupr:{:.4f},f1_score:{:.4f},accuracy:{:.4f},recall:{:.4f},specificity:{:.4f},precision:{:.4f}'.format(metrics[0],
                                                                                                                            metrics[1],
                                                                                                                            metrics[2],
                                                                                                                            metrics[3],
                                                                                                                            metrics[4],
                                                                                                                            metrics[5],
                                                                                                                            metrics[6]))
