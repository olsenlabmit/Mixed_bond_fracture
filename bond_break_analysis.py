import numpy as np
import matplotlib
from matplotlib import pyplot as plt
##matplotlib.use('qtagg')
##set MPLBACKEND=qtagg
##np.seterr(all="warn")
import warnings
##np.testing.suppress_warnings()
runs=[1,10,11,12,13,14,15,16,17,18,19,20,2,3,4,5,6,7,8,9]
##runs=[1,10]
num_runs=len(runs);
count=0
final_weak_broken=np.zeros(11)
final_strong_broken=np.zeros(11)
final_weak_broken_std_dev=np.zeros(11)
final_strong_broken_std_dev=np.zeros(11)
weak_frac_list=[0,10,20,30,40,50,60,70,80,90,100]
for weak_frac in weak_frac_list:#,30,40,50,60,70,80,90,100]:
##    print('count',count)
    avg_weak_broken=0
    avg_strong_broken=0
    i=0
    weak_broken_diff_runs=np.zeros(num_runs)
    strong_broken_diff_runs=np.zeros(num_runs)
    for run in runs:
        
        
        total_strong_broken=0
        total_weak_broken=0
        max_steps=1200
##        step_list=[i for i in range(10,max_steps,10)]
        
        step_count=0
        f="Run"+str(run)+"\\" + str(weak_frac)+ "\\KMC_stats"
##        with warnings.catch_warnings():
##            warnings.simplefilter("ignore")
        data=np.genfromtxt(f, skip_header=1)
        data_shape=np.shape(data)
        total_weak_broken=sum(data[:,3])
        total_strong_broken=sum(data[:,4])
##        stop
##        print('run',run)
##        print(total_weak_broken)
##        print(total_strong_broken)

        
        weak_bond_broken_vs_steps=np.zeros([num_runs,data_shape[0]])
        strong_bond_broken_vs_steps=np.zeros([num_runs,data_shape[0]])#,data_shape[1]])
        extension=np.zeros(data_shape[0])
        for step in range(0,data_shape[0]):
            weak_bond_broken_vs_steps[i,step]=data[step,3]
            strong_bond_broken_vs_steps[i,step]=data[step,4]
            extension[step]=data[step,0]
            
##            if(np.size(data_shape)>1):
##                num_weak_broken=data[-1,9]
##                num_strong_broken=data[-1,10]
##                shape=np.shape(data)
##                total_bonds_broken=shape[0]
##                bond_broken_vs_steps[step_count]=total_bonds_broken                      
##                total_weak_broken=total_weak_broken+num_weak_broken
##                total_strong_broken=total_strong_broken+num_strong_broken
##            step_count=step_count+1
##        stop
        weak_broken_diff_runs[i]=total_weak_broken
        strong_broken_diff_runs[i]=total_strong_broken
##        print(weak_broken_diff_runs)
##        print(strong_broken_diff_runs)                             
        i=i+1
    
    final_weak_broken[count]=np.mean(weak_broken_diff_runs)
    final_weak_broken_std_dev[count]=np.std(weak_broken_diff_runs)
    final_strong_broken[count]=np.mean(strong_broken_diff_runs)
    final_strong_broken_std_dev[count]=np.std(strong_broken_diff_runs)
##    plt.figure()
    plt.plot(extension,np.mean(weak_bond_broken_vs_steps, axis=0),label=str(weak_frac)+'weak')
    plt.plot(extension,np.mean(strong_bond_broken_vs_steps,axis=0),label=str(weak_frac)+'strong')
    plt.legend()
    plt.xlabel('Bond breaking step')
    plt.ylabel('Number of bonds broken in each step')
    count=count+1
strong_frac_list=[100-x for x in weak_frac_list]
plt.figure()


plt.errorbar(strong_frac_list,final_weak_broken[0: len(strong_frac_list)],yerr=final_weak_broken_std_dev,fmt = 'o',color = 'red', 
            ecolor = 'red', elinewidth = 2,capsize=5,label='weak')
##
plt.errorbar(strong_frac_list,final_strong_broken[0: len(strong_frac_list)],yerr=final_strong_broken_std_dev,fmt = 'o',color = 'blue', 
            ecolor = 'blue', elinewidth = 2,capsize=5,label='strong')
total=final_weak_broken[0: len(strong_frac_list)]+final_strong_broken[0: len(strong_frac_list)]
##plt.plot(strong_frac_list,final_weak_broken[0: len(strong_frac_list)],'o-',label='weak')
##
##plt.plot(strong_frac_list,total,'o-',label='strong')
plt.errorbar(strong_frac_list,total,yerr=final_strong_broken_std_dev+final_weak_broken_std_dev,fmt = '-o',color = 'black', 
            ecolor = 'black', elinewidth = 2,capsize=5,label='total')
plt.xticks(fontsize=12,fontweight='bold')
plt.yticks(fontsize=12,fontweight='bold')
##plt.plot(strong_frac_list,total,'ko-',label='total')
print(total)
plt.legend(loc=2,fontsize='xx-large')
plt.xlabel('Percent strong bonds',fontsize=13,fontweight="bold")
plt.ylabel('Number of broken bonds',fontsize=13,fontweight="bold")
plt.savefig('plot_bond_broken')
np.savetxt('data_bond_broken_global.txt', np.transpose([strong_frac_list, final_weak_broken, final_weak_broken_std_dev, final_strong_broken,final_strong_broken_std_dev,total]),header='[strong_frac_list, final_weak_broken, final_weak_broken_std_dev, final_strong_broken,final_strong_broken_std_dev,total]')
plt.show()
