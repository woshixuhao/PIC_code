'''
This is the PIC method code for 2D cases
Made by Hao Xu
2022.8
'''
from neural_network import ANN, random_data
import numpy as np
import torch
from torch.autograd import Variable
import os
import scipy.io as scio
from torch.optim import lr_scheduler
from neural_network import  *
import heapq
#GPU set
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cpu')

torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)
GPUID="1"



def symbol_derivative(d,GM):
    if d==0:
        result=[GM]
        return result

    if d==1:
        result=[]
        for j in range(len(GM)):
            new = GM.copy()
            if new[j] == 0:
                new.append(new[j] + 1)
                new.pop(j)
            else:
                new[j] += 1
            result.append(new)
        return result
    if d==2:
        result=[]
        GM_new=symbol_derivative(1,GM)
        for gene in GM_new:
            gene_result=symbol_derivative(1,gene)
            for g in gene_result:
                result.append(g)
        return result

    if d==3:
         result = []
         GM_new = symbol_derivative(1, GM)
         for gene in GM_new:
             gene_result = symbol_derivative(2, gene)
             for g in gene_result:
                 result.append(g)
         return result

def translate(Final_genome,Final_left):
    translate_genome=[]
    total=[]
    for i in range(len(Final_genome)):
        result=symbol_derivative(d=Final_left[i][1],GM=Final_genome[i])
        for r in result:
            translate_genome.append(r)
    for t in translate_genome:
        if sorted(t) not in total:
            total.append(sorted(t))
    return total


def translate_value_np(genome,ut,u,ux,uxx,uxxx,uy,uyy,uyyy):
    gene_value=np.zeros([u.shape[0],len(genome)])
    for i in range(len(genome)):
        new_gene_value_x=np.ones([u.shape[0],1])
        new_gene_value_y = np.ones([u.shape[0], 1])
        for j in range(len(genome[i])):
            if genome[i][j]==0:
                new_gene_value_x*=u
                new_gene_value_y *= u
            if genome[i][j]==1:
                new_gene_value_x*=ux
                new_gene_value_y *= uy
            if genome[i][j]==2:
                new_gene_value_x*=uxx
                new_gene_value_y *= uyy
            if genome[i][j]==3:
                new_gene_value_x*=uxxx
                new_gene_value_y *= uyyy
        gene_value[:,i]=new_gene_value_x.reshape(u.shape[0])+new_gene_value_y.reshape(u.shape[0])
    return gene_value

def translate_value_cuda(genome,ut,u,ux,uxx,uxxx,uy,uyy,uyyy):
    gene_value=torch.zeros([u.shape[0],len(genome)]).cuda()
    for i in range(len(genome)):
        new_gene_value_x=torch.ones([u.shape[0],1]).cuda()
        new_gene_value_y = torch.ones([u.shape[0], 1]).cuda()
        for j in range(len(genome[i])):
            if genome[i][j] == 0:
                new_gene_value_x *= u
                new_gene_value_y *= u
            if genome[i][j] == 1:
                new_gene_value_x *= ux
                new_gene_value_y *= uy
            if genome[i][j] == 2:
                new_gene_value_x *= uxx
                new_gene_value_y *= uyy
            if genome[i][j] == 3:
                new_gene_value_x *= uxxx
                new_gene_value_y *= uyyy
        gene_value[:,i]=new_gene_value_x.reshape(u.shape[0])+new_gene_value_y.reshape(u.shape[0])
    return gene_value

class PINNLossFunc(nn.Module):
    def __init__(self,h_data_choose):
        super(PINNLossFunc,self).__init__()
        self.h_data=h_data_choose
        return


    def forward(self,prediction,H_t,coef,iter,Library):
        res=H_t
        f1=torch.pow((prediction-self.h_data),2).mean()
        for i in range(Library.shape[1]):
            res=res-Library[:,i].reshape(Library.shape[0],1)*coef[i]
        f2=torch.pow(res,2).mean()

        kesi = 0.01

        MSE=f1 + kesi*f2

        return MSE


def get_sub_set(nums):
    sub_sets = [[]]
    for i in range(len(nums)):
        x=nums[i]
        sub_sets.extend([item + [x] for item in sub_sets])
    sub_sets.pop(0)
    return sub_sets

def calculate_cv(gene_translate):
    coef_array=np.zeros([window_num,len(gene_translate)])
    for k in range(window_num):
        nx=20
        nt=20
        ny=20
        x=torch.linspace(x_low,x_up,nx)
        y=torch.linspace(y_low,y_up,ny)
        dk=(t_up-t_low)/20
        t=torch.linspace(t_low+dk*k,(t_up+t_low)/2+dk*k,nt)
        dx=x[1]-x[0]
        dt=t[1]-t[0]
        dy=y[1]-y[0]
        total=nx*nt*ny
        num=0
        data=torch.zeros(3)
        h_data=torch.zeros([total,1])
        database=torch.zeros([total,3])
        for j in range(nx):
            for m in range(ny):
                for i in range(nt):
                    data[0]=x[j]
                    data[1]=y[m]
                    data[2]=t[i]
                    database[num]=data
                    num+=1

        database_choose = Variable(database, requires_grad=True).to(DEVICE)
        prediction=Net(database_choose)

        H_grad = torch.autograd.grad(outputs=prediction.sum(), inputs=database_choose, create_graph=True)[0]
        Hx = H_grad[:, 0].reshape(total, 1)
        Hy = H_grad[:, 1].reshape(total, 1)
        Ht = H_grad[:, 2].reshape(total, 1)
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database_choose, create_graph=True)[0][:, 0].reshape(total, 1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database_choose, create_graph=True)[0][:, 0].reshape(total,
                                                                                                                  1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database_choose, create_graph=True)[0][:, 1].reshape(total,
                                                                                                                1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database_choose, create_graph=True)[0][:, 1].reshape(total,
                                                                                                                  1)


        # ------terms-----------
        Ht_n = Ht.cpu().data.numpy()
        H_n = prediction.cpu().data.numpy()
        Hx_n = Hx.cpu().data.numpy()
        Hxx_n = Hxx.cpu().data.numpy()
        Hxxx_n = Hxxx.cpu().data.numpy()
        Hy_n = Hy.cpu().data.numpy()
        Hyy_n = Hyy.cpu().data.numpy()
        Hyyy_n = Hyyy.cpu().data.numpy()

        del Hx
        del Ht
        del H_grad
        del Hxx
        del Hxxx
        del Hy
        del Hyy
        del Hyyy
        del prediction
        del database_choose

        Library = translate_value_np(gene_translate, Ht_n, H_n, Hx_n, Hxx_n, Hxxx_n, Hy_n,Hyy_n,Hyyy_n)
        u, d, v = np.linalg.svd(np.hstack((Ht_n, Library)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        lst=coef_NN[1:]
        coef_array[k]=lst.reshape(lst.shape[0])

    cv=[]
    for i in range(coef_array.shape[1]):
        cv.append(np.abs(np.std(coef_array[:,i])/np.mean(coef_array[:,i])))
    #print(coef_array)
    return cv


def look_regression(target,Ht,Htt,u_meta,H,Hx,Hxx,Hxxx,Hxxxx,left='u_t'):
    Ht=Ht.cpu().data.numpy()
    Htt=Htt.cpu().data.numpy()
    u_meta=u_meta.cpu().data.numpy()
    H=H.cpu().data.numpy()
    Hx=Hx.cpu().data.numpy()
    Hxx=Hxx.cpu().data.numpy()
    Hxxx=Hxxx.cpu().data.numpy()
    Hxxxx=Hxxxx.cpu().data.numpy()
    Right=torch.zeros([Ht.shape[0],len(target)])
    for i in range(len(target)):
        new=torch.ones([Ht.shape[0],1])
        for gene in target[i]:
            if gene==0:
                new*=H
            if gene==1:
                new*=Hx
            if gene==2:
                new*=Hxx
            if gene==3:
                new*=Hxxx
            if gene==4:
                new*=Hxxxx
        Right[:,i]=new.reshape(Ht.shape[0])
    if left=='u_t':
        u, d, v = np.linalg.svd(np.hstack((Ht, Right)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        #res_u_t = np.dot(np.hstack((Ht, Right)), coef_NN)
        print(coef_NN)


    if left=='u_tt':
        u, d, v = np.linalg.svd(np.hstack((Htt, Right)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        #res_u_t = np.dot(np.hstack((Htt, Right)), coef_NN)
        print(coef_NN)

    u, d, v = np.linalg.svd(np.hstack((u_meta, Right)), full_matrices=False)
    coef_phi = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
    #res_u_t = np.dot(np.hstack((u_meta, Right)), coef_phi)
    print(coef_phi)

    return coef_NN,coef_phi

class GGA():
    def __init__(self,x,y,t,u,u_x,u_xx,u_xxx,u_y,u_yy,u_yyy,u_t,u_tt,epi,dim=2,delete_num=0,creterion='standard'):
        self.dim=dim
        self.max_length = 3
        self.partial_prob = 0.6
        self.genes_prob = 0.6
        self.mutate_rate = 0.4
        self.delete_rate = 0.5
        self.distiction_rate=0.1
        self.add_rate = 0.4
        self.pop_size = 400
        self.n_generations = 200
        self.delete_num=delete_num
        self.u=u
        self.u_x=u_x
        self.u_xx=u_xx
        self.u_xxx=u_xxx
        self.u_y=u_y
        self.u_yy=u_yy
        self.u_yyy=u_yyy
        self.u_t=u_t
        self.u_tt=u_tt
        self.x=x
        self.t=t
        self.y=y
        self.dx=x[1]-x[0]
        self.dt=t[1]-t[0]
        self.dy=y[1]-y[0]
        self.nx=x.shape[0]
        self.nt=t.shape[0]
        self.ny=y.shape[0]
        self.total_delete=(x.shape[0]-delete_num)*(t.shape[0]-delete_num)*(y.shape[0]-delete_num)
        self.epi=epi
        self.creterion=creterion

    def Delete_boundary(self,u):
        un=u.reshape(self.nt,self.ny,self.nx)
        un_del=un[self.delete_num:self.nt-self.delete_num,self.delete_num:self.ny-self.delete_num,
               self.delete_num:self.nx-self.delete_num]
        return un_del.reshape((self.nx-self.delete_num*2)*(self.nt-self.delete_num*2)*(self.ny-self.delete_num*2),1)

    def FiniteDiff_x(self,un,d):
        #u=[nx,nt]
        #用二阶微分计算d阶微分，不过在三阶以上准确性会比较低
        #u是需要被微分的数据
        #dx是网格的空间大小
        u=un.T
        dx=self.dx
        nt,nx=u.shape
        ux=np.zeros([nt,nx])

        if d==1:

            ux[:,1:nx-1]=(u[:,2:nx]-u[:,0:nx-2])/(2*dx)
            ux[:,0]=(-3.0/2*u[:,0]+2*u[:,1]-u[:,2]/2)/dx
            ux[:,nx-1]=(2.0/2*u[:,nx-1]-2*u[:,nx-2]+u[:,nx-3]/2)/dx
            return  ux.T

        if d==2:
            ux[:,1:nx-1]=(u[:,2:nx]-2*u[:,1:nx-1]+u[:,0:nx-2])/dx**2
            ux[:,0]=(2*u[:,0]-5*u[:,1]+4*u[:,2]-u[:,3])/dx**2
            ux[:,nx-1]=(2*u[:,nx-1]-5*u[:,nx-2]+4*u[:,nx-3]-u[:,nx-4])/dx**2
            return ux.T

        if d==3:
            ux[:,2:nx-2]=(u[:,4:nx]/2-u[:,3:nx-1]+u[:,1:nx-3]-u[:,0:nx-4]/2)/dx**3
            ux[:,0]=(-2.5*u[:,0]+9*u[:,1]-12*u[:,2]+7*u[:,3]-1.5*u[:,4])/dx**3
            ux[:,1]=(-2.5*u[:,1]+9*u[:,2]-12*u[:,3]+7*u[:,4]-1.5*u[:,5])/dx**3
            ux[:,nx-1]=(2.5*u[:,nx-1]-9*u[:,nx-2]+12*u[:,nx-3]-7*u[:,nx-4]+1.5*u[:,nx-5])/dx**3
            ux[:,nx-2]=(2.5*u[:,nx-2]-9*u[:,nx-3]+12*u[:,nx-4]-7*u[:,nx-5]+1.5*u[:,nx-6])/dx**3
            return ux.T

        if d>3:
            return GGA.FiniteDiff_x(GGA.FiniteDiff_x(u,dx,3),dx,d-3)

    def random_diff_module(self):
        if self.dim==1:
            diff_y=0
            diff_x = random.randint(0, 3)
        if self.dim==2:
            diff_y=0
            diff_x=0
        genes_module = [diff_y,diff_x]
        return genes_module

    def random_module(self):
        genes_module=[]
        genes_diff_module = GGA.random_diff_module(self)
        if self.creterion=='standard':
            for i in range(self.max_length):
                a=random.randint(0,2)
                genes_module.append(a)
                prob=random.uniform(0,1)
                if prob>self.partial_prob:
                    break
        if self.creterion=='Galieo':
            a = random.randint(1, 2)
            genes_module.append(a)
        return genes_module,genes_diff_module

    def random_genome(self):
        genes=[]
        gene_diff=[]

        if self.creterion=='Galieo':
            genes.append([0,0])
            gene_diff.append([0,1])

        for i in range(self.max_length):
            gene_random,gene_random_diff=GGA.random_module(self)
            genes.append(sorted(gene_random))
            gene_diff.append((gene_random_diff))
            prob=random.uniform(0,1)
            if prob>self.genes_prob:
                break
        return genes,gene_diff

    def translate_DNA(self,gene,gene_left):
        gene_translate=np.ones([self.total_delete,1])
        length_penalty_coef=0
        for k in range(len(gene)):
            gene_module=gene[k]
            gene_left_module=gene_left[k]
            length_penalty_coef+=1
            module_out_x=np.ones([*u.shape])
            module_out_y = np.ones([*u.shape])
            for i in gene_module:
                if i==0:
                    temp_x=self.u
                if i==1:
                    temp_x=self.u_x
                if i==2:
                    temp_x=self.u_xx
                if i==3:
                    temp_x=self.u_xxx
                module_out_x*=temp_x
            for i in gene_module:
                if i==0:
                    temp_y=self.u
                if i==1:
                    temp_y=self.u_y
                if i==2:
                    temp_y=self.u_yy
                if i==3:
                    temp_y=self.u_yyy
                module_out_y*=temp_y
            un=(module_out_x+module_out_y).reshape(self.nt,self.ny,self.nx)
            if gene_left_module[1]>0:
                un_x=GGA.FiniteDiff_x(self,un,d=gene_left_module[1])
                un=un_x
            un = GGA.Delete_boundary(self,un)
            module_out=un.reshape([self.total_delete,1])
            gene_translate=np.hstack((gene_translate,module_out))
        gene_translate=np.delete(gene_translate,[0],axis=1)
        return gene_translate,length_penalty_coef

    def get_fitness(self,gene_translate,length_penalty_coef):
        u_t=self.u_t
        u_t_new=u_t.reshape([self.nt,self.ny,self.nx])
        u_t=GGA.Delete_boundary(self,u_t_new).reshape(self.total_delete,1)
        u, d, v = np.linalg.svd(np.hstack((u_t, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef=-coef_NN[1:].reshape(coef_NN.shape[0]-1,1)
        res = u_t-np.dot(gene_translate,coef)

        u_tt=self.u_tt
        u_tt_new=u_tt.reshape([self.nt,self.ny,self.nx])
        u_tt=GGA.Delete_boundary(self,u_tt_new).reshape(self.total_delete,1)
        u, d, v = np.linalg.svd(np.hstack((u_tt, gene_translate)), full_matrices=False)
        coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        coef_tt=-coef_NN[1:].reshape(coef_NN.shape[0]-1,1)
        res_tt = u_tt-np.dot(gene_translate,coef_tt)
        MSE_true = np.sum(np.array(res) ** 2) / self.total_delete
        MSE_true_tt = np.sum(np.array(res_tt) ** 2) / self.total_delete

        if MSE_true<MSE_true_tt:
            name='u_t'
            MSE=MSE_true+self.epi*length_penalty_coef
            coef=coef
            return coef,MSE,MSE_true,name


        if MSE_true>MSE_true_tt:
            name='u_tt'
            MSE = MSE_true_tt + self.epi * length_penalty_coef
            return coef_tt, MSE, MSE_true_tt, name



    def cross_over(self):
        Chrom,Chrom_diff, size_pop = self.Chrom,self.Chrom_diff, self.n_generations
        Chrom1, Chrom2 = Chrom[::2], Chrom[1::2]
        Chrom1_diff, Chrom2_diff = Chrom_diff[::2], Chrom_diff[1::2]
        for i in range(int(size_pop / 2)):
            if self.creterion=='standard':
                n1= np.random.randint(0, len(Chrom1[i]))
                n2=np.random.randint(0, len(Chrom2[i]))

            if self.creterion=='Galieo':
                if len(Chrom1[i])>1 and len(Chrom2[i])>1:
                    n1 = np.random.randint(1, len(Chrom1[i]))
                    n2 = np.random.randint(1, len(Chrom2[i]))
                else:
                    continue


            father=Chrom1[i][n1].copy()
            mother=Chrom2[i][n2].copy()

            father_diff=Chrom1_diff[i][n1].copy()
            mother_diff=Chrom2_diff[i][n2].copy()

            Chrom1[i][n1]=mother
            Chrom2[i][n2]=father

            Chrom1_diff[i][n1] = mother_diff
            Chrom2_diff[i][n2] = father_diff

        Chrom[::2], Chrom[1::2] = Chrom1, Chrom2
        Chrom_diff[::2], Chrom_diff[1::2] = Chrom1_diff, Chrom2_diff
        self.Chrom = Chrom
        self.Chrom_diff = Chrom_diff
        return self.Chrom,self.Chrom_diff

    def mutation(self):
        Chrom,Chrom_diff, size_pop = self.Chrom,self.Chrom_diff, self.pop_size

        for i in range(size_pop):
            n1 = np.random.randint(0, len(Chrom[i]))

            # ------------add module---------------
            prob = np.random.uniform(0, 1)
            if prob < self.add_rate:
                add_Chrom,add_Chrom_diff = GGA.random_module(self)
                if add_Chrom not in Chrom[i]:
                    Chrom[i].append(add_Chrom)
                    Chrom_diff[i].append(add_Chrom_diff)

            # --------delete module----------------
            prob = np.random.uniform(0, 1)
            if prob < self.mutate_rate:
                if len(Chrom[i]) > 1:
                    if self.creterion=='standard':
                        delete_index = np.random.randint(0, len(Chrom[i]))
                    if self.creterion=='Galieo':
                        delete_index = np.random.randint(1, len(Chrom[i]))
                    Chrom[i].pop(delete_index)
                    Chrom_diff[i].pop(delete_index)

            # ------------gene mutation------------------
            if self.creterion=='standard':
                prob = np.random.uniform(0, 1)
                if prob < self.mutate_rate:
                    if len(Chrom[i]) > 0:
                        n1 = np.random.randint(0, len(Chrom[i]))
                        n2 = np.random.randint(0, len(Chrom[i][n1]))
                        Chrom[i][n1][n2] = random.randint(0,3)
                        Chrom_diff[i][n1]=GGA.random_diff_module(self)


        self.Chrom = Chrom
        self.Chrom_diff=Chrom_diff
        return self.Chrom,self.Chrom_diff


    def select(self):  # nature selection wrt pop's fitness
        Chrom, Chrom_diff,size_pop = self.Chrom,self.Chrom_diff, self.pop_size
        new_Chrom=[]
        new_fitness=[]
        new_Chrom_diff=[]
        new_coef=[]
        new_name=[]

        fitness_list = []
        coef_list=[]
        name_list=[]

        for i in range(size_pop):
            gene_translate, length_penalty_coef = GGA.translate_DNA(self, Chrom[i],Chrom_diff[i])
            coef, MSE, MSE_true,name = GGA.get_fitness(self, gene_translate, length_penalty_coef)
            fitness_list.append(MSE)
            coef_list.append(coef)
            name_list.append(name)
        re1 = list(map(fitness_list.index, heapq.nsmallest(int(size_pop/2), fitness_list)))
        for index in re1:
            new_Chrom.append(Chrom[index])
            new_Chrom_diff.append(Chrom_diff[index])
            new_fitness.append(fitness_list[index])
            new_coef.append(coef_list[index])
            new_name.append(name_list[index])
        for index in range(int(size_pop/2)):
            new,new_diff=GGA.random_genome(self)
            new_Chrom.append(new)
            new_Chrom_diff.append(new_diff)


        self.Chrom=new_Chrom
        self.Chrom_diff=new_Chrom_diff
        self.Fitness=new_fitness
        self.coef=new_coef
        self.name=new_name
        return self.Chrom,self.Fitness,self.coef,self.name

    def delete_duplicates(self):
        Chrom,Chrom_diff, size_pop = self.Chrom,self.Chrom_diff, self.pop_size
        for i in range(size_pop):
            new_genome=[]
            new_genome_diff=[]
            for j in range(len(Chrom[i])):
                if sorted(Chrom[i][j]) not in new_genome:
                    new_genome.append(sorted(Chrom[i][j]))
                    new_genome_diff.append(Chrom_diff[i][j])
            Chrom[i]=new_genome
            Chrom_diff[i]=new_genome_diff
        self.Chrom=Chrom
        self.Chrom_diff=Chrom_diff
        return self.Chrom,self.Chrom_diff

    def disctinction(self):
        Chrom, Chrom_diff, size_pop = self.Chrom, self.Chrom_diff, self.pop_size
        prob = np.random.uniform(0, 1)
        if prob < self.distiction_rate:
            new_Chrom=[]
            new_Chrom_diff=[]
            for iter in range(self.pop_size):
                intial_genome, intial_genome_diff = GGA.random_genome(self)
                new_Chrom.append(intial_genome)
                new_Chrom_diff.append(intial_genome_diff)
            self.Chrom=Chrom
            self.Chrom_diff=Chrom_diff

    def evolution(self):
        self.Chrom = []
        self.Chrom_diff=[]
        self.Fitness=[]
        for iter in range(self.pop_size):
            intial_genome,intial_genome_diff =GGA.random_genome(self)
            self.Chrom.append(intial_genome)
            self.Chrom_diff.append(intial_genome_diff)
            gene_translate, length_penalty_coef=GGA.translate_DNA(self,intial_genome,intial_genome_diff)
            coef, MSE,MSE_true,name=GGA.get_fitness(self,gene_translate,length_penalty_coef)
            self.Fitness.append(MSE)
        GGA.delete_duplicates(self)
        for iter in range(self.n_generations):
            print(f'--------{iter}----------------')
            np.save('../best_save.npy', np.array(self.Chrom.copy()[0]), allow_pickle=True)
            np.save('../best_save_diff.npy', np.array(self.Chrom_diff.copy()[0]), allow_pickle=True)
            best =self.Chrom.copy()[0]
            best_nc=self.Chrom_diff.copy()[0]
            GGA.cross_over(self)
            GGA.mutation(self)
            #GGA.disctinction(self)
            GGA.delete_duplicates(self)
            best = np.load('../best_save.npy', allow_pickle=True).tolist()
            best_diff = np.load('../best_save_diff.npy', allow_pickle=True).tolist()
            self.Chrom[0]=best
            self.Chrom_diff[0]=best_diff
            GGA.select(self)
            print(f'The best Chrom: {self.Chrom[0]}')
            print(f'The best diff:  {self.Chrom_diff[0]}')
            print(f'The best coef:  \n{self.coef[0]}')
            print(f'The best fitness: {self.Fitness[0]}')
            print(f'The best name: {self.name[0]}\r')

        return self.Chrom[0],self.Chrom_diff[0],self.coef[0],self.Fitness[0],self.name[0]

def Generate_meta_data_2D(Net,Equation_name, choose, noise_level, trail_num, Load_state, x_low, x_up, t_low, t_up, nx=20,
                       nt=20,ny=20):
    Net.load_state_dict(torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
    Net.eval()

    x = torch.linspace(x_low, x_up, nx)
    t = torch.linspace(t_low, t_up, nt)
    y=torch.linspace(y_low,y_up,ny)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    dy=y[1]-y[0]
    total = nx * nt*ny
    total_delete = (nx - delete_num) * (nt - delete_num)*(ny-delete_num)

    num = 0
    data = torch.zeros(3)
    h_data = torch.zeros([total, 1])
    database = torch.zeros([total, 3])
    for j in range(nx):
        for k in range(ny):
            for i in range(nt):
                data[0] = x[j]
                data[1]=y[k]
                data[1] = t[i]
                database[num] = data
                num += 1

    database = Variable(database, requires_grad=True).to(DEVICE)
    PINNstatic = Net(database)
    H_grad = torch.autograd.grad(outputs=PINNstatic.sum(), inputs=database, create_graph=True)[0]
    Hx = H_grad[:, 0].reshape(total, 1)
    Hy = H_grad[:, 1].reshape(total, 1)
    Ht = H_grad[:, 2].reshape(total, 1)
    Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database, create_graph=True)[0][:, 0].reshape(total, 1)
    Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database, create_graph=True)[0][:, 0].reshape(total, 1)
    Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database, create_graph=True)[0][:, 1].reshape(total, 1)
    Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database, create_graph=True)[0][:, 1].reshape(total, 1)
    Htt = torch.autograd.grad(outputs=Ht.sum(), inputs=database, create_graph=True)[0][:, 2].reshape(total, 1)

    # ----------convert to numpy-----------------
    H_n = PINNstatic.cpu().data.numpy()
    Hx_n = Hx.cpu().data.numpy()
    Hxx_n = Hxx.cpu().data.numpy()
    Hxxx_n = Hxxx.cpu().data.numpy()
    Hy_n = Hy.cpu().data.numpy()
    Hyy_n = Hyy.cpu().data.numpy()
    Hyyy_n = Hyyy.cpu().data.numpy()
    Ht_n = Ht.cpu().data.numpy()
    Htt_n = Htt.cpu().data.numpy()

    Theta = H_n
    Theta = np.hstack((Theta, Hx_n))
    Theta = np.hstack((Theta, Hxx_n))
    Theta = np.hstack((Theta, Hxxx_n))
    Theta = np.hstack((Theta, Hy_n))
    Theta = np.hstack((Theta, Hyy_n))
    Theta = np.hstack((Theta, Hyyy_n))
    Theta = np.hstack((Theta, Ht_n))
    Theta = np.hstack((Theta, Htt_n))
    np.save("Theta-GA", Theta)

    return Theta,x.data.numpy(),y.data.numpy(),t.data.numpy()

#============Params=============
Equation_name='Burgers_2D'
choose=200000
noise_level=50
noise_type='Gaussian' #Gaussian or Uniform
trail_num='PIS'
Learning_Rate=0.001
Activation_function="Rational" #'Tanh','Rational'
criterion='standard'
#============Get origin data===========
if Equation_name=='Burgers_2D':
    data_path= f'data/{Equation_name}/Burgers2D.mat'
    Data=scio.loadmat(data_path)
    x = np.squeeze(Data["x"])
    y = np.squeeze(Data["y"])
    t = np.squeeze(Data["t"])
    un = Data["u"]
    x_low = -0.8
    x_up = 0.8
    t_low = 0
    t_up = 2
    y_low=-0.8
    y_up=0.8
    target=[[2],[0,1]]
    Left = 'u_t'
    epi = 1e-6
    #epi=1e-5


print(x)
print(y)
print(t)
print(un.shape)
#x和T
x_num=x.shape[0]
y_num=y.shape[0]
t_num=t.shape[0]
print(x.shape,y.shape,t.shape,un.shape)
total=x_num*t_num*y_num
print(total)
choose_validate=5000
meta_data_num=10000
delete_num = 0
window_num=10
if noise_type=='Uniform':
    for j in range(t_num):
        for i in range(y_num):
            for k in range(x_num):
                un[j,i,k]=un[j,i,k]*(1+0.01*noise_level*np.random.uniform(-1,1))
if noise_type=='Gaussian':
    noise_value=(noise_level/100)*np.std(un)*np.random.randn(*un.shape)
    un=un+noise_value
#save model dir

try:
    os.makedirs(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})')
except OSError:
    pass

try:
    os.makedirs(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})')
    np.save(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}', un)
except OSError:
    un=np.load(f'noise_data_save/{Equation_name}/{choose}_{noise_level}({noise_type})/un_{noise_level}.npy')
    print('===load noisy data===')
    pass



#=========produce random dataset==========
h_data_choose,h_data_validate,database_choose,database_validate=random_data_2D(total,choose,choose_validate,x,y,t,un,x_num,y_num,t_num)
database_choose = Variable(database_choose.cuda(),requires_grad=True)
database_validate = Variable(database_validate.cuda(),requires_grad=True)
h_data_choose=Variable(h_data_choose.cuda())
h_data_validate=Variable(h_data_validate.cuda())

h_max=np.max(h_data_choose.cpu().data.numpy())
h_min=np.min(h_data_choose.cpu().data.numpy())
print('difference:',h_max-h_min)
#==========NN setting=============
torch.manual_seed(525)
torch.cuda.manual_seed(525)
#Net=ANN(2,50,1).to(DEVICE)
Net=NN(Num_Hidden_Layers=5,
    Neurons_Per_Layer=50,
    Input_Dim=3,
    Output_Dim=1,
    Data_Type=torch.float32,
    Device=DEVICE,
    Activation_Function=Activation_function,
    Batch_Norm=False)

NN_optimizer = torch.optim.Adam([
    {'params': Net.parameters()},
])


MSELoss = torch.nn.MSELoss()
best_validate_error=[]
loss_back = 1e8
flag=0
print(f'===============train Net=================')
for iter in range(30000):
    NN_optimizer.zero_grad()
    prediction = Net(database_choose)
    prediction_validate = Net(database_validate).cpu().data.numpy()
    loss = MSELoss(h_data_choose, prediction)
    loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
    loss.backward()
    NN_optimizer.step()



    if (iter+1) % 500 == 0:
        # if loss_validate.item() < loss_back:
        #     loss_back = loss_validate.item()
        # else:
        #     flag+=1
        #     if flag==3:
        #         best_validate_error.append(loss_back)
        #         break
        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (iter+1, loss, loss_validate))
        torch.save(Net.state_dict(), f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/'+f"Net_{Activation_function}_{iter+1}.pkl")



#=======================Generalized GA====================
# min_value = min(best_validate_error)
# best_activation_index=best_validate_error.index(min_value)
#activation_name=Activation_function[best_activation_index]
Load_state='Net_'+Activation_function+'_30000'
print(best_validate_error)
print(f'Use {Load_state}')

R,x_meta,y_meta,t_meta=Generate_meta_data_2D(Net,Equation_name,choose,noise_level,trail_num,Load_state,x_low,x_up,t_low,t_up)
u=R[:,0].reshape(R.shape[0],1)
u_x=R[:,1].reshape(R.shape[0],1)
u_xx=R[:,2].reshape(R.shape[0],1)
u_xxx=R[:,3].reshape(R.shape[0],1)
u_y=R[:,4].reshape(R.shape[0],1)
u_yy=R[:,5].reshape(R.shape[0],1)
u_yyy=R[:,6].reshape(R.shape[0],1)
u_t=R[:,7].reshape(R.shape[0],1)
u_tt=R[:,8].reshape(R.shape[0],1)

Theta=np.squeeze(np.array([u*u_x+u*u_y,u_xx+u_yy])).T
print(np.linalg.lstsq(Theta,u_t)[0])
error=u_t+(u*u_x+u*u_y)-0.01*(u_xx+u_yy)
print(np.mean(np.abs(error)))
print(np.sum(np.abs(u_t)))
gga=GGA(x_meta,y_meta,t_meta,u,u_x,u_xx,u_xxx,u_y,u_yy,u_yyy,u_t,u_tt,epi=epi,dim=2,delete_num=delete_num,creterion=criterion)
gene_translate=gga.translate_DNA([[0,1],[2]],[[0,0],[0,0]])[0]
print(gga.get_fitness(gene_translate,2))
gene_translate=gga.translate_DNA([[1],[2]],[[0,0],[0,0]])[0]
print(gga.get_fitness(gene_translate,2))
Chrom,Chrom_diff,coef,Fitness,best_name=gga.evolution()
print('===============Finish!==============')
print(f'The potential genome is {Chrom}\nThe corresponding diff is {Chrom_diff}\n The coef is {coef.reshape(coef.shape[0])}\n The fitness is {Fitness}'
      f'\n The best name is {best_name}')




#========================Variance drop and PINN=========================
print('\n================variance drop start!================\n')
Net.load_state_dict(torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
genome=translate(Chrom,Chrom_diff)
sub_sets=get_sub_set(genome)

cv_mean=[]
for i in range(len(sub_sets)):
    cv=calculate_cv(sub_sets[i])
    cv_mean.append(np.mean(np.array(cv)))
    print(f'The subset is:  {sub_sets[i]}\nThe cv is:  {cv}\nThe mean is:  {np.mean(np.array(cv))}')
    print('-------------------------------------------')
re1 = list(map(cv_mean.index, heapq.nsmallest(5, cv_mean)))

for i in range(len(re1)):
    index=re1[i]
    print(f'The #{i+1} possible PDE is {sub_sets[index]}  with cv mean  {cv_mean[index]}')

if Activation_function=='Sin':
    x_num=20
    y_num=20
    t_num=20
if Activation_function=='Rational':
    x_num=20
    y_num=20
    t_num=20
total=x.shape[0]*t.shape[0]
#produce random dataset
test_num=x_num*t_num*y_num
x_test = torch.linspace(x_low, x_up, x_num)
y_test=torch.linspace(y_low, y_up, y_num)
t_test = torch.linspace(t_low, t_up, t_num)
num = 0
data = torch.zeros(3)
database_test = torch.zeros([test_num, 3])
for j in range(x_num):
    for k in range(y_num):
        for i in range(t_num):
            data[0] = x_test[j]
            data[1]=y_test[k]
            data[2] = t_test[i]
            database_test[num] = data
            num += 1

database_test = Variable(database_test, requires_grad=True).to(DEVICE)

#neural network

num=0
torch.manual_seed(525)

#NN
MSELoss = torch.nn.MSELoss()
Error=[]
Error_origin=[]
potential_PDE=[]
prediction_test_compare = Net(database_test).cpu().data.numpy()
for index in re1:
    print(f'============PINN training start for {sub_sets[index]}!=============')
    Net.load_state_dict(
        torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))

    optimizer = torch.optim.Adam([
        {'params': Net.parameters()},
    ])
    gene_translate=sub_sets[index]
    for t in range(300):
        optimizer.zero_grad()
        prediction = Net(database_choose)
        prediction_validate = Net(database_validate).cpu().data.numpy()
        prediction_test = Net(database_test)

        H_grad = torch.autograd.grad(outputs=prediction_test.sum(), inputs=database_test, create_graph=True)[0]
        Hx = H_grad[:, 0].reshape(test_num, 1)
        Hy = H_grad[:, 1].reshape(test_num, 1)
        Ht = H_grad[:, 2].reshape(test_num, 1)
        Htt = torch.autograd.grad(outputs=Ht.sum(), inputs=database_test, create_graph=True)[0][:, 2].reshape(test_num,
                                                                                                              1)
        Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database_test, create_graph=True)[0][:, 0].reshape(test_num,
                                                                                                              1)
        Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database_test, create_graph=True)[0][:, 0].reshape(
            test_num,1)
        Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database_test, create_graph=True)[0][:, 1].reshape(test_num,
                                                                                                              1)
        Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database_test, create_graph=True)[0][:, 1].reshape(
            test_num,1)

        # ------terms-----------
        Ht_n = Ht.cpu().data.numpy()
        H_n = prediction_test.cpu().data.numpy()
        Hx_n = Hx.cpu().data.numpy()
        Hxx_n = Hxx.cpu().data.numpy()
        Hxxx_n = Hxxx.cpu().data.numpy()
        Hy_n=Hy.cpu().data.numpy()
        Hyy_n = Hyy.cpu().data.numpy()
        Hyyy_n = Hyyy.cpu().data.numpy()

        Library=translate_value_np(gene_translate,Ht_n,H_n,Hx_n,Hxx_n,Hxxx_n,Hy_n,Hyy_n,Hyyy_n)
        Library_cuda=translate_value_cuda(gene_translate,Ht,prediction_test,Hx,Hxx,Hxxx,Hy,Hyy,Hyyy)
        u, d, v = np.linalg.svd(np.hstack((Ht_n, Library)), full_matrices=False)
        if t<=300:
            coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
        lst=-coef_NN[1:].reshape(coef_NN.shape[0]-1,1)
        coef = Variable(torch.from_numpy(lst.astype(np.float32))).cuda()
        a = PINNLossFunc(h_data_choose)
        loss = a(prediction, Ht, coef, t,Library_cuda)
        loss.backward()
        optimizer.step()
        loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
        if (t + 1) % 100 == 0:
            print(coef)
            print(f'iter_num: {t+1}/{300}')
    error=np.sqrt(np.mean((Net(database_test).cpu().data.numpy()-prediction_test_compare)**2))
    #error=np.mean(np.abs((prediction-h_data_choose).cpu().data.numpy()))
    print(
        f'The PDE is {sub_sets[index]},  The coef is {coef.cpu().data.numpy().reshape(coef.shape[0])},  Error: {error}\n')

    potential_PDE.append(sub_sets[index])
    Error_origin.append(error)
    Error.append(error*cv_mean[index])

Error_origin=np.array(Error_origin)
cv_Error=np.abs(np.std(Error_origin)/np.mean(Error_origin))
print(cv_Error)
min_value = min(Error)
best_PDE_index=Error.index(min_value)
best_PDE=potential_PDE[best_PDE_index]
for i in range(len(potential_PDE)):
    print(f'Error of {potential_PDE[i]}    :    {Error[i]}')

# for i in range(len(Error)):
#     if (min_value-Error[i])/Error[i]<0.01 and (min_value-Error[i])/Error[i]>0:
#         if len(best_PDE)>len(potential_PDE[i]):
#             min_value=Error[i]
#             best_PDE_index=i
#             best_PDE=potential_PDE[i]




print(f'The best PDE is {best_PDE}')
print(f'========Lets train best PDEs coef by PINN==========')
Net.load_state_dict(
    torch.load(f'model_save/{Equation_name}/{choose}_{noise_level}_{trail_num}({noise_type})/{Load_state}.pkl'))
prediction_test = Net(database_test)
optimizer = torch.optim.Adam([
    {'params': Net.parameters()},
])
for t in range(1000):
    gene_translate = best_PDE
    optimizer.zero_grad()
    prediction = Net(database_choose)
    prediction_validate = Net(database_validate).cpu().data.numpy()
    prediction_test = Net(database_test)

    H_grad = torch.autograd.grad(outputs=prediction_test.sum(), inputs=database_test, create_graph=True)[0]
    Hx = H_grad[:, 0].reshape(test_num, 1)
    Hy = H_grad[:, 1].reshape(test_num, 1)
    Ht = H_grad[:, 2].reshape(test_num, 1)
    Htt = torch.autograd.grad(outputs=Ht.sum(), inputs=database_test, create_graph=True)[0][:, 2].reshape(test_num,
                                                                                                          1)
    Hxx = torch.autograd.grad(outputs=Hx.sum(), inputs=database_test, create_graph=True)[0][:, 0].reshape(test_num,
                                                                                                          1)
    Hxxx = torch.autograd.grad(outputs=Hxx.sum(), inputs=database_test, create_graph=True)[0][:, 0].reshape(
        test_num, 1)
    Hyy = torch.autograd.grad(outputs=Hy.sum(), inputs=database_test, create_graph=True)[0][:, 1].reshape(test_num,
                                                                                                          1)
    Hyyy = torch.autograd.grad(outputs=Hyy.sum(), inputs=database_test, create_graph=True)[0][:, 1].reshape(
        test_num, 1)

    # ------terms-----------
    Ht_n = Ht.cpu().data.numpy()
    H_n = prediction_test.cpu().data.numpy()
    Hx_n = Hx.cpu().data.numpy()
    Hxx_n = Hxx.cpu().data.numpy()
    Hxxx_n = Hxxx.cpu().data.numpy()
    Hy_n = Hy.cpu().data.numpy()
    Hyy_n = Hyy.cpu().data.numpy()
    Hyyy_n = Hyyy.cpu().data.numpy()

    Library = translate_value_np(gene_translate, Ht_n, H_n, Hx_n, Hxx_n, Hxxx_n, Hy_n, Hyy_n, Hyyy_n)
    Library_cuda = translate_value_cuda(gene_translate, Ht, prediction_test, Hx, Hxx, Hxxx, Hy, Hyy, Hyyy)
    u, d, v = np.linalg.svd(np.hstack((Ht_n, Library)), full_matrices=False)
    coef_NN = v.T[:, -1] / (v.T[:, -1][0] + 1e-8)
    lst = -coef_NN[1:].reshape(coef_NN.shape[0] - 1, 1)

    coef = Variable(torch.from_numpy(lst.astype(np.float32))).cuda()
    a = PINNLossFunc(h_data_choose)
    loss = a(prediction, Ht, coef,1e8 , Library_cuda)
    loss.backward()
    optimizer.step()
    loss_validate = np.sum((h_data_validate.cpu().data.numpy() - prediction_validate) ** 2) / choose_validate
    if (t + 1) % 1000 == 0:
        print(f'iter_num: {t+1}/{3000},  coef: {coef.cpu().data.numpy().reshape(coef.shape[0])}')

print(f'The best PDE is {best_PDE}\n The best coef is {coef.cpu().data.numpy().reshape(coef.shape[0])}')