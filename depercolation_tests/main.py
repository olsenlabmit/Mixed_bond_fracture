#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

"""
#######################################
#                                     #
#-- Fracture Simulation of Networks --#
##Depercolation tests##
Updated Devosmita Sen February 2023
#                                     #
#######################################

 Overall Framework (Steps):
     1. Generate a Network following the algorithm published
        by AA Gusev, Macromolecules, 2019, 52, 9, 3244-3251
        if gen_net = 0, then it reads topology from user-supplied 
        network.txt file present in this folder
     
     2. Force relaxtion of network using Fast Inertial Relaxation Engine (FIRE) 
        to obtain the equilibrium positions of crosslinks (min-energy configuration)

     3. Compute Properties: Energy, Gamma (prestretch), and 
        Stress (all 6 componenets) 
     
     4. Deform the network (tensile) in desired direction by 
        strain format by supplying lambda_x, lambda_y, lambda_z

     5. Break bonds using Kintetic Theory of Fracture (force-activated KMC) 
        presently implemented algorithm is ispired by 
        Termonia et al., Macromolecules, 1985, 18, 2246

     6. Repeat steps 2-5 until the given extension (lam_total) is achived OR    
        stress decreases below a certain (user-specified) value 
        indicating that material is completey fractured.
"""
import os.path
import sys
#file_dir = os.path.dirname(__file__)
#sys.path.append(file_dir)
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#sys.path.append('../')
#print(sys.path)

import time
import math
import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
##import ioLAMMPS
##import netgen
##from relax import Optimizer
from numpy import linalg as LA
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import param as p
import shutil
import random



direction=0 #z
 
'''
###########parameters##################
U0_low=1
U0_high=5

N_low=12.0
N_high=12.0

##frac_weak=0.0

L=30 # corresponding to loop frac~=0.05
lam_max=10
del_t=0.01
e_rate=5
n_chains=500

b_low=1.0
b_high=1.0

K_low=1.0
K_high=1.0

fit_param_low=1.0
fit_param_high=1.0

E_b_low=1200.0
E_b_high=1200.0

func=4

tol=0.01
max_itr = 100000
write_itr = 10000
wrt_step = 500
##################################################
'''
#random.seed(a=500)
random.seed(a=None, version=2)
##random.seed(10)
##random.seed(10)
print('First random number of this seed: %d'%(random.randint(0, 10000))) 
# This is just to check whether different jobs have different seeds
##global parameters
parameters=np.zeros([2,6]) # N, b, K, fit_param, E_b,U0
#parameters[0,:]=np.array([12,1.0,1.0,1.0,1200,56.7578])
#parameters[1,:]=np.array([12,1.0,1.0,1.0,1200,283.789])
##parameters[0,:]=np.array([p.N_low,p.b_low,p.K_low,p.fit_param_low,p.E_b_low,p.U0_low])
##parameters[1,:]=np.array([p.N_high,p.b_high,p.K_high,p.fit_param_high,p.E_b_high,p.U0_high])


parameters[0,:]=np.array([p.N_low,p.b_low,p.K_low,p.fit_param_low,p.E_b_low,p.U0_low])
parameters[1,:]=np.array([p.N_high,p.b_high,p.K_high,p.fit_param_high,p.E_b_high,p.U0_high])

##frac_weak=0.0
##frac_weak=p.frac_weak
#%#non deterministic step for assignning chain_type
##my_task_id = int(sys.argv[1])
##num_tasks = int(sys.argv[2])
frac_weak_array_py=[0.0]#,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
##frac_weak_array_py=frac_weak_array_py[my_task_id-1:len(frac_weak_array_py):num_tasks]      

for frac_weak in frac_weak_array_py:
   '''
   directory = './'+str(int(100*frac_weak))+'/'
   file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory) 
   '''
   directory = './original_files/'
   orig_dir = os.path.dirname(directory)
   files=os.listdir(orig_dir)
   directory = './'+str(int(100*frac_weak))+'/'
   if not os.path.isdir(directory):
      os.mkdir(directory)

   for fname in files:
     
    # copying the files to the
    # destination directory
       shutil.copy2(os.path.join(orig_dir,fname), directory)

# now add path to frac_weak directory
   file_dir = os.path.dirname(directory)
   sys.path.append(file_dir)
   import ioLAMMPS
   import netgen
   from relax import Optimizer
   import networkx as nx
   
   # in this, which parameter is assigned is given by variable chain_type

   netgen_flag = 1
   swell = 0
   if(netgen_flag==0):

      vflag = 0
   ##   N = 12   
      print('--------------------------')   
      print('----Reading Network-------')   
      print('--------------------------')
      
      filename = "network.txt"
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
      [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
              atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS(file_path, vflag, frac_weak)
      
      print('xlo, xhi',xlo, xhi) 
      print('ylo, yhi',ylo, yhi) 
      print('zlo, zhi',zlo, zhi) 
      print('n_atoms', n_atoms) 
      print('n_bonds', n_bonds) 
      print('atom_types = ', atom_types) 
      print('bond_types = ', bond_types) 
      print('mass = ', mass) 
      print('primary loops = ', len(loop_atoms)) 
      print('--------------------------')   

   elif(netgen_flag==1):

   ##   func = 4
      func=p.func
   ##   N    = 12
   ##   rho  = 3
      l0   = 1
      prob = 1.0
      #n_chains=10000
      n_chains  = p.n_chains
      n_links   = int(2*n_chains/func)
      #L = 28
      L=p.L
      print(prob, func, parameters,L, l0, n_chains, n_links)
      G=nx.MultiGraph()
      linker_to_node=netgen.generate_network(G,prob, func, parameters,L, l0, n_chains, n_links, frac_weak)
      directory = './'+str(int(100*frac_weak))+'/'
      filename = 'network.txt'
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
      
      [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
              atom_types, bond_types, mass, loop_atoms] = ioLAMMPS.readLAMMPS(file_path,0,frac_weak)

   ##   print('atoms \n',atoms)
   ##   print('bonds \n',bonds)
   ##   stop
   else:
      print('Invalid network generation flag')

   ##stop
            
##   save_path =str(100*int(frac_weak))
##   completeName = os.path.join(save_path,"stress")         
## 
   directory = './'+str(int(100*frac_weak))+'/'
   filename = 'stress'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   fstr=open(file_path,'w')
   fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5]\n') 

   filename = 'strand_lengths'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)      
   flen=open(file_path,'w')
   flen.write('#lambda, ave(R), max(R)\n') 


   filename = 'KMC_stats'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)
   fkmc=open(file_path,'w')
   fkmc.write('#lambda, init bonds, final bonds, weak_bonds_broken,strong_bonds_broken\n') 
   #-------------------------------------#
   #       Simulation Parameters         #
   #-------------------------------------#

   #N  = 12
   ##chain_type=
   ##Nb = N
   ##K  = 1.0
   r0 = 0.0
   #U0  = 1 # not used anywhere inside the functions, just passed to function as parameters
   tau = 1# not used anywhere inside the functions, just passed to function as parameters
   #del_t=0.008
   del_t=p.del_t
   erate = p.e_rate
   #lam_max = 25
   lam_max = p.lam_max
   ##tol = 0.01
   ##max_itr = 100000
   ##write_itr = 10000
   ##wrt_step = 500
   tol = p.tol
   max_itr = p.max_itr
   write_itr = p.write_itr
   wrt_step = p.wrt_step

   G_orig=G.copy()

  
   
#####################################################################
   # finding correct values of delta_x right and left for applying voltage and getting current
   delta_x_left=0.5
   delta_x_right=0.5
   '''
   min_count=50.0
   left_count=0
   right_count=0
##   for i in G:
####      while(left_count<min_count):
##      if(atoms[i,0]>=x_left and atoms[i,0]<=x_left+3.0):
##         print('i',i,'x-position',atoms[i,0])
####   stop
   
   while(left_count<min_count):
      left_count=0
      for i in G:
         if(atoms[i,0]>=x_left and atoms[i,0]<=x_left+delta_x_left ):
                  left_count=left_count+1
##                  print('i',i,'x-position',atoms[i,0])
##                  print('left_count',left_count,'delta_x_left',delta_x_left)
##                  print('delta_x_left',delta_x_left,'x-pos',atoms[i,0])
      if(left_count<min_count):
         delta_x_left=delta_x_left+1.0
   print('delta_x_left',delta_x_left)
##   stop
##   for i in G:
####      while(left_count<min_count):
##      if(atoms[i,0]>=x_left and atoms[i,0]<=x_left+delta_x_left):
##         print('i',i,'x-position',atoms[i,0])

         
   while(right_count<min_count):
      right_count=0
      for i in G: 
         if(atoms[i,0]<=x_right and atoms[i,0]>=x_right-delta_x_right):
                  right_count=right_count+1
      if(right_count<min_count):
         delta_x_right=delta_x_right+1.0
   print('delta_x_right',delta_x_right)


   #####################################################################
   '''

##   G=nx.Graph()
##   G.clear()
##   G.add_edge(1,2)
##   stop            
   print('Stopping before force relaxation')
   # left boundary
##   x=# x-coordinate of crosslinker
   potential_node_1_list=[]
##   count=0
##   for i in G:
##    if(i in G.neighbors(i)):
##        count=count+1
##   print('G.number_of_edges()',G.number_of_edges())
##   print('count',count)
##   print('n_bonds',n_bonds)
##   stop

##   for i in G:
####         print('Hello 1')
##         if(len(list(G.neighbors(i)))>0):
##            potential_node_1_list.append(i)

##   output_current_array=[]
##   R_eff=[]
##   R_tot=[]
##   V_l=[]
##   V_r=[]
   I_in_arr=[]
   I_out_arr=[]
   density=[]
   max_connected=[]
   second_largest_connected=[]
   avg_cluster_size_without_largest=[]
   path_exists_array=[]
   num_paths_array=[]
   fraction_paths_connected=[]
##   frac_cleaved_array=[]
   
   frac_cleaved_array=np.zeros(1)
   frac_cleaved_array=np.append(frac_cleaved_array,np.linspace(0.1,0.4,10))
   frac_cleaved_array=np.append(frac_cleaved_array,np.linspace(0.401,0.6,50))
   frac_cleaved_array=np.append(frac_cleaved_array,np.linspace(0.61,0.9,10))
##   frac_cleaved_array=[0.0,0.1,0.2,0.3,0.34,0.37,0.4,0.44,0.48,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.8,0.9]
   for frac_cleaved in frac_cleaved_array: # frac_cleaved-low- current will be there
##      actual_potential_node_1=potential_node_1
      G_temp=G.copy()
      # beyond a certain fraction cleaved- we will not get any current
##      link_1=
      
##      stop
##         if(list(G.neighbors(i))==[]):
##            print('something wrong')
##            stop
##      stop
##            potential_node_1.append(i)
##      node_1_coutn=0
##      while(node_1_count<=frac_cleaved*
##      for j in np.arrange(len(potential_node_1)):
##      node_1_list=random.sample(actual_potential_node_1, int(frac_cleaved*len(potential_node_1)))
##      for node_1 in node_1_list:
####         print('Hello3')
##         if(len(list(G.neighbors(node_1)))==0):
##            print('PROBLEM!!')
##            stop
##         node_2_list=list(G.neighbors(node_1))
##         node_2=random.choice(node_2_list)
##         G.remove_edge(node_1,node_2)
##      print('Initially: number of egdes:',G_temp.number_of_edges())
      init_num_bonds=G_temp.number_of_edges()
##      stop
      num_broken_bonds=0
      
##      print('number of egdes:',G_temp.number_of_edges())
      while(num_broken_bonds<int(frac_cleaved*init_num_bonds)):
########         for node_1 in potential_node_1_list:
########            if(num_broken_bonds<int(frac_cleaved*init_num_bonds)):
########   ##         while(num_broken_bonds<int(frac_cleaved*init_num_bonds)):
########   ##            if(list(G.neighbors(node_1))!=[]): # always check whether bond is broken on the fly- can't do beforehnd
########   ##              if next(G.neighbors(node_1),None) is not None:
########               a=list(nx.all_neighbors(G_temp, node_1))
########               if(a!=[]):
########   ##               print('node_1',node_1)
########                  node_2_list=list(G_temp.neighbors(node_1))
########                  node_2=random.choice(node_2_list)
   ##               print('list of neighbors before breaking',list(nx.all_neighbors(G, node_1)))
                  rnd=random.randrange(len(G_temp.edges))
                  node_1=list(G_temp.edges)[rnd][0]
                  node_2=list(G_temp.edges)[rnd][1]
                  G_temp.remove_edge(node_1,node_2)
                  num_broken_bonds=num_broken_bonds+1
               
##               print('list of neighbors after breakign',list(nx.all_neighbors(G, node_1)))
##               print('number of egdes after breaking:',G.number_of_edges())
##
##               stop
      density.append(nx.density(G_temp))# density is being calculated before breaking the periodic boundary condition
      
      max_connected.append(max(len(cc) for cc in nx.connected_components(G_temp)))
      cc=list(nx.connected_components(G_temp))
      cc.sort(key=len)
      second_largest_connected.append(len(cc[len(cc)-2]))# this is being calculated before breaking the periodic bc
##      avg_cluster_size_without_largest.append(np.mean([len(x) for x in cc[0:-1]]))
      all_cluster_sizes=[len(x) for x in cc[0:-1]] # without largest
      cluster_sizes, number_distribution=np.unique(all_cluster_sizes,return_counts=True)
##           stop
##           number_distribution=np.zeros(np.shape(cluster_sizes))
      numerator=0
      denominator=0
      for i in range(len(cluster_sizes)):
         numerator=numerator+(cluster_sizes[i]**2)*number_distribution[i]
         denominator=denominator+(cluster_sizes[i])*number_distribution[i]
               
      avg_cluster_size_without_largest.append(numerator/denominator)#np.mean([len(x) for x in cc[0:-1]]))
           
      sys_size=len(G_temp)
      I=np.zeros(sys_size) # current array- 1D
##      delta_x=3.0


      #######################################################################
       # breaking periodic boundary condition:
#IGNORE THIS COMMENT: THIS IS NOT PROPERLY BREAKING THE BOUNDARY CONDITION!!- WILL HAVE TO SEE THE DISTANCES LIKE I DID EARLIER
      delta_x=15.0#L/2
      x_left=0.0
      x_right=30.0
      I_val=1.0
      nodes_left=[]
      nodes_right=[]
      nodes_int=[] # intermediate nodes
##      for i in G_temp:
##         if(atoms[i,0]>=x_left and atoms[i,0]<=x_left+delta_x):
##            for j in list(G_temp.neighbors(i)):
##               if(atoms[j,0]<=x_right and atoms[j,0]>=x_right-delta_x):  # i can do this here because i know that there will be no connection
##                  # which is going through middle and satifies this consition as well
##                  # because this is the inital network and there are no stretched chains here- max chain lenfth can be the contour lenfth
##                  G_temp.remove_edge(i,j) # break bonds which are connected through periodic bc
##                  I[i]=I_val
##                  I[j]=-1.0*I_val
      cnt_length=12.0
     
      for i in G_temp:
##         if(atoms[i,0]>=x_left and atoms[i,0]<=x_left+delta_x):
         for j in list(G_temp.neighbors(i)):
            chain_length=np.sqrt((atoms[i,0]-atoms[j,0])**2+(atoms[i,1]-atoms[j,1])**2+(atoms[i,2]-atoms[j,2])**2)
            if(chain_length>cnt_length and (atoms[i,direction]-15.0)*(atoms[j,direction]-15.0)<0.0):#abs(atoms[i,0]-atoms[j,0])>=18.0): # this means that connection is through the other direction (through L--> 0 end)(periodic bc)
               # i can do this here because i know that there will be no connection
                  # which is going through middle and satifies this consition as well
                  # because this is the inital network and there are no stretched chains here- max chain lenfth can be the contour lenfth
                  G_temp.remove_edge(i,j) # break bonds which are connected through periodic bc
##                  I[i]=I_val
##                  I[j]=-1.0*I_val
                  x_i=atoms[i,direction]
                  x_j=atoms[j,direction]
##                  print('i',atoms[i,:],'j',atoms[j,:],'chain_length',chain_length)
                  if(x_i>=x_left and x_i<=x_left+delta_x):
##                     I[i]=I_val
                     nodes_left.append(i)
                     
##                     continue
##                     print('x_i',x_i,'x_j',x_j,'chain_length',chain_length)
                  if(x_j>=x_left and x_j<=x_left+delta_x):
##                     I[j]=I_val
                     nodes_left.append(j)
                     
##                     continue
##                     print('x_j',x_j,'x_i',x_i,'chain_length',chain_length)
                  if(x_i<=x_right and x_i>=x_right-delta_x):
##                     I[i]=-1.0*I_val
                     nodes_right.append(i)
                     
##                     continue
##                     print('x_i',x_i,'x_j',x_j,'chain_length',chain_length)
                  if(x_j<=x_right and x_j>=x_right-delta_x):
##                     I[j]=-1.0*I_val
                     nodes_right.append(j)
##                  stop
##                     continue
##                     print('x_j',x_j,'x_i',x_i,'chain_length',chain_length)
##      V_needed=np.zeros(sys_size)# crosslinkers for which current will have to be calcylated
      
##      count_l=0
##      count_r=0
##      matrix_count=0
##      V0=10.0
##      V_max=V0
##      V_min=-1.0*V0
##      I_trunc=[]
##      V_trunc=[]
##      node_index=[]
##
##      print('nodes_left',nodes_left)
##      print('nodes_right',nodes_right)
      #####################################################################
      print('number of nodes on left boundary',len(nodes_left))
      print('number of nodes on right boundary',len(nodes_right))

      path_exists=0
      num_paths=0
      for l in nodes_left:
         for r in nodes_right:
            if(nx.has_path(G_temp,l,r)):
               path_exists=1
               break
      for l in nodes_left:
         for r in nodes_right:
            if(nx.has_path(G_temp,l,r)):
               num_paths=num_paths+1
      path_exists_array.append(path_exists)
      num_paths_array.append(num_paths)
##      fraction_paths_connected.append(num_paths/(len(nodes_left)*len(nodes_right)))
      #len(nodes_left)*len(nodes_right) is the total number of paths that should have been
      #there if every node on left was connected to every node on right!
##      stop
##      stop
      for i in G_temp: # current and voltage calculation should be separate!!
         x=atoms[i,direction]#x_coordinate of crosslinker i
         if(x>=x_left and x<=x_left+delta_x_left):
##            V_needed_l.append(i)
##            V_trunc.append(V_max)
##            node_index.append(i)
##            matrix_count=matrix_count+1
##            nodes_left.append(i)
            continue
##            count_l=count_l+1
      for i in G_temp: # current and voltage calculation should be separate!!
         x=atoms[i,direction]#x_coordinate of crosslinker i
         if(x<=x_right and x>=x_right-delta_x_right):
##            V_trunc.append(V_min)
##            node_index.append(i)
##            matrix_count=matrix_count+1
##            nodes_right.append(i)
            continue
##            count_r=count_r+1
##
      for i in G_temp:
         if(i not in nodes_right and i not in nodes_left):
            nodes_int.append(i)
##      V_needed_l=[]
##      V_needed_r=[]

      A1=np.zeros((sys_size,sys_size))
      A2=np.zeros((sys_size,sys_size))
      B1=np.zeros(sys_size)
      B2=np.zeros(sys_size)

      V0=1.0
      V_l=V0
      V_r=-1.0*V0
      I_int=0.0

      for i in G_temp:
         if i in nodes_int:
            A1[i,i]=1.0
            B2[i]=I_int
         elif i in nodes_left:
            A2[i,i]=-1.0
            B1[i]=V_l
         elif i in nodes_right:
            A2[i,i]=-1.0
            B1[i]=V_r
      L = nx.laplacian_matrix(G_temp)# I=LV
      A=np.add(np.matmul(L.toarray(),A1),A2)
      B=np.add(np.matmul(L.toarray(),B1),B2)
      X=LA.lstsq(A,np.multiply(B,-1.0))[0] # contains current for nodes on boundaries, and voltage for intermediate nodes
      I_in=0
      I_out=0
##      for i in nodes_left:
##         I_in=I_in+X[i]
##      for i in nodes_right:
##         I_out=I_out+X[i]
      for i in range(0,len(X)):
         if i not in nodes_int:
            if(X[i]>=0):
               I_in=I_in+X[i]
            else:
               I_out=I_out+X[i]
      I_in_arr.append(I_in)
      I_out_arr.append(I_out)
##      L = nx.laplacian_matrix(G_temp)
##      e = np.linalg.eigvals(L.A)
##      L_trunc=np.zeros((matrix_count,matrix_count)) # truncated to contain only the nodes on the boundaries
##      for i in range(0,matrix_count):
##         node_i=node_index[i]
##         for j in range(0,matrix_count):
##            node_j=node_index[j]
##            L_trunc[i][j]=L[node_i,node_j]
##      V_trunc=np.array(V_trunc)
##      I_trunc=np.matmul(L_trunc,V_trunc)
##
##      I_input.append(sum(I_trunc[0:count_l]))
##      I_output.append(sum(I_trunc[count_l:matrix_count]))
        
##      R_tot_i=0
##      e_count=0
##      for i in e:
##         if(i!=0):
##            R_tot_i=R_tot_i+(1.0/i)
##            e_count=e_count+1

##      R_tot=R_tot*e_count
      '''
      L_inv=np.linalg.pinv(L.toarray())
      V_all=np.matmul(L_inv,I) #L_inv
      sumV_l=0
      sumV_r=0
      for i in V_needed_l:
          sumV_l=sumV_l+V_all[i]
      for i in V_needed_r:
          sumV_r=sumV_r+V_all[i]
      R_i=(sumV_l/len(V_needed_l)-sumV_r/len(V_needed_r))/I_val # considering average voltage
      R_tot.append(R_i)
      V_l.append(sumV_l)
      V_r.append(sumV_r)
      '''
##      print('R_i',R_i)
##      stop
##      I_all=np.matmul(np.transpose(V),L.toarray())#np.matmul(V,L_inv)
##      I_final=np.zeros(sys_size)
##      for i in range(sys_size):
##         I_final[i]=I_needed[i]*I_all[i]
##      np.savetxt('current.txt',I_final)
      print('frac_cleaved',frac_cleaved)
      print('Initial total bonds',init_num_bonds)
      print('number of broken_bonds',num_broken_bonds)
      print('Final total bonds',G_temp.number_of_edges())
##      print('Resistance',R_i)
##      print('input voltage',sum(V))
##      print('output current',sum(I_final))
      print('DONE!!')
##      output_current_array.append(sum(I_final))
##      R_eff_i=0
##      for i in range(sys_size):
##         for j in range(sys_size):
##            if(i!=j):
##               R_eff_i=R_eff_i+L_inv[i][j]
##      R_eff.append(np.sum(L_inv))
##      R_eff.append(R_eff_i)
##      R_tot.append(R_tot_i)
   print('delta_x_left',delta_x_left)
   print('delta_x_right',delta_x_right)
##   plt.plot(frac_cleaved_array,output_current_array,'bo-')
   plt.plot(frac_cleaved_array,np.array(path_exists_array),'bo-')#,label='input current')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Does path exist?')
   plt.savefig('does_path_exist.png')

##   plt.figure()
##   plt.plot(frac_cleaved_array,np.array(num_paths_array),'bo-')#,label='input current')
##   plt.xlabel('Fraction cleaved')
##   plt.ylabel('Number of paths')

   '''plt.figure()
   plt.plot(frac_cleaved_array,np.array(fraction_paths_connected),'bo-')#,label='input current')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Number of paths/Total possible paths between nodes on left and right')
   plt.savefig('fraction_of_possible_paths_connected.png')
   '''
   plt.figure()
   plt.plot(frac_cleaved_array,density,'bo-')#,label='input current')
##   plt.plot(frac_cleaved_array,I_in_arr,'bo-',label='input current')
##   plt.plot(frac_cleaved_array,np.abs(I_out_arr),'ro-',label='output current')
##   
##   plt.plot(np.array(frac_cleaved_array), sum(V)*np.ones(len(frac_cleaved_array)),'r--',label='Voltage')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Density')
   plt.savefig('density.png')
   plt.legend()

   plt.figure()
   plt.plot(frac_cleaved_array,max_connected,'bo-')#,label='input current')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Size of largest connected component')
   plt.savefig('size_largest_connected_component.png')

   plt.figure()
   plt.plot(frac_cleaved_array,second_largest_connected,'bo-')#,label='input current')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Size of SECOND largest connected component')
   plt.savefig('size_second_largest_connected_component.png')

   plt.figure()
   plt.plot(frac_cleaved_array, avg_cluster_size_without_largest,'bo-')#,label='input current')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Average cluster size without largest')
   plt.savefig('avg_cluster_size_withiut_largest.png')
   plt.figure()
   plt.plot(frac_cleaved_array,np.abs(I_out_arr),'ro-',label='output current')
   plt.plot(frac_cleaved_array,np.abs(I_in_arr),'bo-',label='intput current')
   plt.legend()
   plt.savefig('current.png')
   m,b = np.polyfit(frac_cleaved_array[0:3], I_in_arr[0:3], 1)
   m1=m
   b1=b
   I_fit_1=np.multiply(frac_cleaved_array[0:6],m)+b
   m,b = np.polyfit(frac_cleaved_array[8:11], I_in_arr[8:11], 1)
   m2=m
   b2=b
   I_fit_2=np.multiply(frac_cleaved_array[3:11],m)+b
   plt.plot(frac_cleaved_array[0:6],I_fit_1,'k--')
   plt.plot(frac_cleaved_array[3:11],I_fit_2,'k--')
   print('Transition at:',(b2-b1)/(m1-m2))
##   plt.title('abs(I_output-I_input)/I_input')
##   plt.plot(np.array(frac_cleaved_array), sum(V)*np.ones(len(frac_cleaved_array)),'r--',label='Voltage')
   plt.xlabel('Fraction cleaved')
   plt.ylabel('Current')
##   plt.figure()
##   plt.plot(frac_cleaved_array,abs(np.divide(np.subtract(I_out_arr,I_in_arr),I_in_arr)),'ro-')#,label='input current')
##   plt.xlabel('Fraction cleaved')
##   plt.ylabel('abs(I_output-I_input)/I_input')
####   plt.ylabel('Resistance')
##   plt.legend()
   np.savetxt('all_data_percolation.txt',np.transpose(np.array([frac_cleaved_array,np.array(path_exists_array),np.array(num_paths_array),
                                                                density,max_connected, I_in_arr,I_out_arr,avg_cluster_size_without_largest,second_largest_connected])),header='[frac_cleaved_array,np.array(path_exists_array),np.array(num_paths_array),density,max_connected, I_in_arr,I_out_arr,avg_cluster_size_without_largest,second_largest_connected]')

   plt.show()
   stop
   #-------------------------------------#
   #       First Force Relaxation        #
   #-------------------------------------#

   mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, r0, parameters, 'Mao')
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
   directory = './'+str(int(100*frac_weak))+'/'
   if(swell==1):
      
      filename = 'restart_network_01.txt'
      file_path = os.path.join(directory, filename)
      if not os.path.isdir(directory):
         os.mkdir(directory)  
   
      ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
      # Swelling the network to V = 2
      scale_x = 1.26
      scale_y = 1.26
      scale_z = 1.26
      mymin.change_box(scale_x, scale_y, scale_z)    
      [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')


   filename = 'restart_network_0.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory)  
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

   dist = mymin.bondlengths()
   Lx0 = mymin.xhi-mymin.xlo
   BE0 = e
   [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
   fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                             %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
   fstr.flush()

   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
   flen.flush()

   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds, n_bonds,0,0))
   fkmc.flush()


   filename = 'restart_network_0.txt'
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory) 
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                     mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

   stop

   
   #-------------------------------------#
   # Tensile deformation: lambda/scales  #
   #-------------------------------------#
##   sys.pause()
   steps = int((lam_max-1)/(erate*del_t))
   print('Deformation steps = ',steps)
   begin_break = -1         # -1 implies that bond breaking begins right from start
   #begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation

   for i in range(0,steps):

       scale_x = (1+(i+1)*erate*del_t)/(1+i*erate*del_t)
       scale_y = scale_z = 1.0/math.sqrt(scale_x)
       mymin.change_box(scale_x, scale_y, scale_z)    
       [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log.txt')
       [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
       fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                        %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                     (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
       fstr.flush()

       dist = mymin.bondlengths()
       flen.write('%7.4fn'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
       flen.flush()
      
       if((i+1)%wrt_step==0): 
         filename = 'restart_network_%d.txt' %(i+1)
         file_path = os.path.join(directory, filename)
         if not os.path.isdir(directory):
            os.mkdir(directory) 
         ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                              mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

       if(i > begin_break):
         # U0, tau, del_t, pflag, index
##         sys.pause()
         [t, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken] = mymin.KMCbondbreak( tau, del_t, 0, i+1,frac_weak)
         if(n_bonds_final<n_bonds_init):
            print('bond broken in function')
##            sys.pause()
         
         fkmc.write('%7.4f  %5i  %5i  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
         fkmc.flush()
    
   #---------------------------------#
   #     Final Network Properties    #
   #---------------------------------#
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, 'log'+str(100*frac_weak)+'.txt')
   [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure()
   fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                    %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                 (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
   fstr.flush()

   dist = mymin.bondlengths()
   flen.write('%7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0))#, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
   flen.flush()

   fkmc.write('%7.4f  %5i  %5i %5i %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final,weak_bond_broken, strong_bond_broken))
   fkmc.flush()
   
   filename = 'restart_network_%d.txt' %(i+1)
   file_path = os.path.join(directory, filename)
   if not os.path.isdir(directory):
      os.mkdir(directory) 
   
   ioLAMMPS.writeLAMMPS(file_path, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                          mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

   fstr.close()
   flen.close()
   fkmc.close()
   sys.path.remove(file_dir)
