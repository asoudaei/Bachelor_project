import numpy as np
import threading
import queue
import matplotlib.pyplot as plt
import gc
queue=queue.Queue()
import os
class MA:
    def __init__(self, obj,
                 npart=10,
                 ndim=3,
                 max_iter=200,
                 tol=None,
                 init=None,
                 done=None,
                 bounds=None,
                 CR=0.5,
                 F=0.05,
                 top=0.5,
                 maxnodes=60,
                 minnodes=30,
                 a=2,
                 runspernet=2):
        self.obj = obj  ## The objective function
        self.npart = npart  ## Number of particles
        self.ndim = ndim  ## Number of input dimensions
        self.max_iter = max_iter  ## Max number of iterations
        self.tol = tol  ## Tolerance value
        self.init = init  ## Initializer type
        self.done = done  ## Done object
        self.bounds = bounds  ## Lower and upper bounds for the input dimensions
        self.CR = CR  ## Crossover probability
        self.F = F  ## Mutation probability
        self.top = top  ## Top particles to breed
        self.Initialized = False
        self.a = a
        self.maxnodes = maxnodes
        self.minnodes = minnodes
        self.worst_val_mapes = []
        self.mean_val_mapes = []
        self.best_val_mapes = []
        self.worst_train_mapes = []
        self.mean_train_mapes = []
        self.best_train_mapes = []
        self.bestcount = 0
        self.runspernet=runspernet



    def Step(self):
        self.Evolve()
        print("ITERATION:", self.iterations)
        self.vpos = self.Evaluate(self.pos)
        self.plotworst(self.worst_val_mapes, self.worst_train_mapes)
        self.plotmean(self.mean_val_mapes, self.mean_train_mapes)
        self.plotbest(self.best_val_mapes, self.best_train_mapes)
        for i in range(self.npart):
            if self.vpos[i] < self.gbest[-1]:
                self.bestcount += 1
                self.gbest.append(self.vpos[i])
                self.models[i].save('BESTMODELMA.h5')
                img_name = "imageMA" + str(i) + ".png"
                new_img_name = "bestmodelimageMA" + str(self.bestcount) + ".png"
                os.rename(img_name, new_img_name)
                print("made" + new_img_name)
                img_name = "2imageMA" + str(i) + ".png"
                new_img_name = "2bestmodelimage" + str(self.bestcount) + ".png"
                os.rename(img_name, new_img_name)
                print("made" + new_img_name)
                self.gpos.append(self.pos[i].copy())
                self.gidx.append(i)
                self.giter.append(self.iterations)
        p=self.vpos
        print("Best MSE so far", self.gbest[-1])
        bestpos = self.gpos[-1]
        bestpos_t = self.transformpositions(np.array([bestpos]))
        print("Best architecture so far:", bestpos_t[0])
        print("Mean MSE this iteration:", np.mean(p))
        print("Best MSE updates:",self.gbest)
        print("######################################################")
        del p
        gc.collect()
        print("Giter:", self.giter)
        self.iterations += 1

    def Evolve(self):
        idx = np.argsort(self.vpos)
        for k, i in enumerate(idx):
            if k == 0:
                continue
            if np.random.random() < self.CR:
                self.Crossover(i, idx)
            if np.random.random() < self.F:
                self.Mutate(i)
            if self.iterations % 2==0:
                self.pos[i]=self.Local_search(i)
            if self.bounds != None:
                self.pos = self.bounds.Limits(self.pos)

    def Mutate(self, idx):
        j = np.random.randint(0, self.ndim)
        if self.bounds != None:
            self.pos[idx, j] = self.bounds.lower[j] + np.random.random() * (self.bounds.upper[j] - self.bounds.lower[j])
        else:
            lower = self.pos[:, j].min()
            upper = self.pos[:, j].max()
            self.pos[idx, j] = lower + np.random.random() * (upper - lower)

    def Crossover(self, a, idx):
        n = int(self.top * self.npart)
        b = idx[np.random.randint(0, n)]
        while a == b:
            b = idx[np.random.randint(0, n)]
        d = np.random.randint(0, self.ndim)
        t = self.pos[a].copy()
        t[d:] = self.pos[b, d:]
        self.pos[a] = t.copy()

    def Initialize(self):
        """Set up the swarm"""

        self.initialized = True
        self.iterations = 0

        self.pos = self.init.InitializeSwarm()  # initial swarm positions
        self.vpos = self.Evaluate(self.pos)  # and objective function values

        #  Swarm bests
        self.gidx = []  ## Swarm ID's that found best positions
        self.gbest = []  ## Global best objection function values
        self.gpos = []  ## Global best positions
        self.giter = []  ## Iterations where global best positions were found
        self.gidx.append(np.argmin(self.vpos))
        self.gbest.append(self.vpos[self.gidx[-1]])
        self.gpos.append(self.pos[self.gidx[-1]].copy())
        self.giter.append(0)
    def Local_search(self,i):
        pos = self.pos[i]
        pos_t = self.transformpositions(np.array([pos]))
        fw = self.obj.Evaluate_simple(pos_t,i)[2]  ## Fitness of the original position
        d = self.a + np.random.random()  ## Generating a random incrementation value
        for k in range(self.ndim):  ## Looping through every dimension
            original_k = self.pos[i,k]  ## Original dimension value
            new_value_k = original_k + d  ## Adding the incrementation value to the original dimension value
            self.pos[i, k] = new_value_k  ## Increasing the position's value in the given dimension
            fn = self.obj.Evaluate_simple(pos_t,i)[2] ## Calculating the new fitness value
            if fn > fw:  ## Checking if we DONT have a better fitness, if NOT:
                new_value_k = original_k - d  ## Subtract the incrementation value from the original dimension value
                self.pos[i, k] = new_value_k  ## Decreasing the position's value in the given dimension
                fn = self.obj.Evaluate_simple(pos_t,i)[2]  ## Calculating the new fitness value
            if fn > fw:  ## If we still don't have a better fitness:
                self.pos[i,k] = original_k  ## We go back to the original dimension value
        return self.pos[i]  ## Return the position

    def Optimize(self):
        self.Initialize()
        while (not self.Done()):
            self.Step()
        bestpos=self.gpos[-1]
        bestpos_t=self.transformpositions(np.array([bestpos]))
        self.gpos[-1]=bestpos_t[0]
        return self.gbest[-1], self.gpos[-1]  ## Returns the LATEST global best value and the LATEST global best position

    def Results(self):
        """Return the current results"""

        if (not self.initialized):
            return None

        return {
            "npart": self.npart,  # number of particles
            "ndim": self.ndim,  # number of dimensions
            "max_iter": self.max_iter,  # maximum possible iterations
            "iterations": self.iterations,  # iterations actually performed
            "tol": self.tol,  # tolerance value, if any
            "gbest": self.gbest,  # sequence of global best function values
            "giter": self.giter,  # iterations when global best updates happened
            "gpos": self.gpos,  # global best positions
            "gidx": self.gidx,  # particle number for new global best
            "pos": self.pos,  # current particle positions
            "vpos": self.vpos,  # and objective function values
        }

    def Evaluate(self,pos):
        p=np.zeros((self.npart))## Creating an array of zeros for each particle
        p2=np.zeros((self.npart))
        self.models=np.zeros((self.npart))
        self.images=np.zeros((self.npart))
        self.models=list(self.models)
        self.images=list(self.images)
        threads=[]
        particle_mapes=[]
        thread_response=[]
        t_pos=self.transformpositions(pos)
        for j in range(self.runspernet):
            print("Round",j+1)
            for i in range(self.npart):## Looping through every particle
                t=threading.Thread(target=self.obj.Evaluate,args=[t_pos[i],queue,i])## Calculating the objection function values at the positions of the particles and putting them in p
                threads.append(t)
            for t in threads:
                t.start()
            for i in range(len(threads)):
                threads[i].join()
                mapes=queue.get()
                particle_mapes.append(mapes)
            threads = []
        new_particle_mapes=[]
        for i in range(self.npart):
            individual_vals=[]
            individual_mapes=[]
            for j in range(len(particle_mapes)):
                mapes=particle_mapes[j]
                if i==mapes[2]:
                   individual_vals.append(mapes[0])
                   individual_mapes.append(mapes)
            k=np.argmin(individual_vals)
            new_particle_mapes.append(individual_mapes[k])
        del particle_mapes
        particle_mapes=new_particle_mapes
        for mapes in particle_mapes:
            p[mapes[2]]=mapes[0]
            p2[mapes[2]] = mapes[1]
            self.models[mapes[2]]=mapes[3]
            history=mapes[4]
            history2=mapes[5]
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model MSE')
            plt.ylabel('MSE')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            imgname = "imageMA" + str(mapes[2])
            plt.savefig(imgname)
            plt.close()
            plt.plot(history2.history['loss'])
            plt.plot(history2.history['val_loss'])
            plt.title('model MSE')
            plt.ylabel('MSE')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            imgname = "2imageMA" + str(mapes[2])
            plt.savefig(imgname)
            plt.close()
            del history
        del threads
        gc.collect()
        self.worst_val_mapes.append(np.max(p))
        self.mean_val_mapes.append(np.mean(p))
        self.best_val_mapes.append(np.min(p))
        self.worst_train_mapes.append(np.max(p2))
        self.mean_train_mapes.append(np.mean(p2))
        self.best_train_mapes.append(np.min(p2))
        return p## Return the objection function values for every particle
    def Done(self):
        if (self.done == None):  ## Runs code below if we are not done
            if (self.tol == None):  ## Checks if we dont have a tolerance
                return (self.iterations == self.max_iter)  ## Return true if we are at the last iteration
            else:
                return (self.gbest[-1] < self.tol) or (
                        self.iterations == self.max_iter)  ## Return true if we have reached a value below the global best
        else:
            return self.done.Done(self.gbest,
                                  gpos=self.gpos,
                                  pos=self.pos,
                                  max_iter=self.max_iter,
                                  iteration=self.iterations)
    def transformpositions(self,pos):
        node_intervals=np.linspace(0,1,self.maxnodes-self.minnodes+2)
        transformed_pos=np.zeros((pos.shape))
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                for k in range(len(node_intervals) - 1):
                    if pos[i,j] >= node_intervals[k] and pos[i,j] < node_intervals[k + 1]:
                        transformed_pos[i,j]=self.minnodes+k
                    if pos[i,j] == 0:
                        transformed_pos[i,j]=self.minnodes
                        break
                    if pos[i,j] == 1:
                        transformed_pos[i,j]=self.maxnodes
        return transformed_pos

    def plotworst(self,val_MSEs, train_MSEs):
        iterations = np.linspace(0, len(val_MSEs)-1, len(train_MSEs))
        plt.plot(iterations, val_MSEs, 'g')
        plt.plot(iterations, train_MSEs, 'r')
        plt.legend(['Validation MSE', 'Train MSE'])
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Worst swarm MSES over generations")
        plt.savefig("figPSO1.png")
        plt.close()

    def plotmean(self,val_MSEs, train_MSEs):
        iterations = np.linspace(0, len(val_MSEs)-1, len(train_MSEs))
        plt.plot(iterations, val_MSEs, 'g')
        plt.plot(iterations, train_MSEs, 'r')
        plt.legend(['Validation MSE', 'Train MSE'])
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Mean swarm MSES over generations")
        plt.savefig("figPSO2.png")
        plt.close()

    def plotbest(self,val_MSEs, train_MSEs):
        iterations = np.linspace(0, len(val_MSEs)-1, len(train_MSEs))
        plt.plot(iterations, val_MSEs, 'g')
        plt.plot(iterations, train_MSEs, 'r')
        plt.legend(['Validation MSE', 'Train MSE'])
        plt.xlabel('Generation')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(iterations), 5))
        plt.title("Best swarm MSES over generations")
        plt.savefig("figPSO3.png")
        plt.close()
