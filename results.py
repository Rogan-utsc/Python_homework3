from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
num = np.arange(40).reshape(10,4)#生成10行4列的数组

dataset_names = ['cifar10', 'cifar100']

def get_curve(full_path):
    with open(full_path) as fp:
        Eval_acc_curve=[]
        Eval_Loss_curve=[]
        lines = fp.readlines()
        for line in lines:
            idx1,idx2 = line.find("Eval Loss:"),line.find("Eval acc: ")
            if idx1 >-1:
                line1, line2 = line[idx1:idx2], line[idx2:]
                Eval_Loss,Eval_acc = float(line1[len("Eval Loss:"):]),float(line2[len("Eval acc: "):])
                Eval_acc_curve.append(Eval_acc)
                Eval_Loss_curve.append(Eval_Loss)
            else:
                continue
        return(Eval_acc_curve,Eval_Loss_curve)

def main():
    Acc_dict_cifar10=dict()
    Acc_dict_cifar100=dict()
    for data_name in dataset_names:
        data_path = os.path.join('../results',data_name)
        path_list = os.listdir(data_path)
        cou = 0
        for path in path_list:
            loss_name = path[:-4]
            exp_name = data_name + "_" + loss_name
            full_path = os.path.join(data_path,path)
            curves = get_curve(full_path)
            if len(curves[0])>0:
                curve1,curve2 =curves[0],curves[1]
            if data_name == "cifar10":
                Acc_dict_cifar10[loss_name]=curve1[-1]
            else:
                Acc_dict_cifar100[loss_name]=curve1[-1]
            if loss_name != "nlnl":
                cou = cou+1
                if cou%2 == 0:
                    linestyle = '-'
                else:
                    linestyle= '--'
                plt.subplot(1,2,1)
                plt.title(data_name)
                plt.plot(range(len(curve1)),scipy.signal.savgol_filter(curve1,21,3)/100,label = loss_name, linestyle = linestyle )
                plt.legend(loc="lower left",fontsize="xx-small")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.subplot(1,2,2)
                plt.title(data_name)
                plt.plot(range(len(curve2)),scipy.signal.savgol_filter(curve2,21,3)/100,label = loss_name)
                plt.legend(loc="upper left",fontsize="xx-small")
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
        #plt.show()
    print(Acc_dict_cifar10)
    print(Acc_dict_cifar100)

if __name__ == '__main__':
    main()



