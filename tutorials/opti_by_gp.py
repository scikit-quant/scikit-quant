from pyDOE import *
from scipy.spatial import distance
import scipy.spatial as scp
import time

from pyDOE import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.gaussian_process.kernels import ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    from qiskit.algorithms.optimizers import IMFIL
except ImportError:
    print("install scikit-quant to use IMFIL")


def obj_wrapper(inp_x,objective,histfilename):
	res=objective(inp_x)
	#f= open(histfilename,"a+") #store parameter values and energy in log file
	#for ii in range(len(inp_x)):
	#	f.write("%f  " % (inp_x[ii]))
	#f.write("%f \n" % (res))
	#f.close()
	return res


def opti_by_gp(dim, bounds,objective, maxevals, histfilename='pointlog.txt'):
	#file where all input-output pairs are stored
	f= open(histfilename, "w+")
	f.close()

	ninit = min(2*(dim+1),maxevals) #number of points in initial experimental design for GP  - tunable
	xlow=bounds[:,0] #lower parameter bounds
	xup=bounds[:,1] #upper parameter bounds
	print(xlow, xup)
	#GP stuff
	#create and evaluate an initial experimental design
	init_design = lhs(dim, samples =ninit, criterion='maximin') #initial design in [0,1]^dim
	Xsamples = xlow+(xup-xlow)*init_design #scale to [xlow,xup]
	Ysamples=np.zeros((ninit,1)) #for storing energies
	for jj in range(ninit): #evaluate each paramter set in initial desing
		Ysamples[jj,0] = objective(Xsamples[jj,:]) #obj_wrapper(Xsamples[jj,:],objective, histfilename) #energy

	n_GP_samples = min(10*(dim+1),maxevals-ninit) #max number of GP-based samples we allow -- tunable
	minID=np.argmin(Ysamples)
	best_point=Xsamples[minID,:] #best parameter values found so far
	best_f = Ysamples[minID,:] #best energy found so far
	#define GP kernel and train GP model - tunable
	kernel = RBF()+WhiteKernel(noise_level=0.05) #white kernel should be used when noise is present
	bound_list = np.concatenate((xlow.reshape(-1,1), xup.reshape(-1,1)), axis = 1)

    #GP iterations: sample my expected improvement
	while  Xsamples.shape[0]< n_GP_samples:
		gpr_obj = GaussianProcessRegressor(kernel=kernel, random_state=0,normalize_y=True, n_restarts_optimizer=10).fit(Xsamples, Ysamples) #create the GP
		#compute next point by maximizing expected improvement
		too_close = True
		success = False
		maxtrials = 5 #try 5 optimizations of expected improvement, if all unsuccessful, use a random point
		n_trials=0
		xnew = []
		fnew = np.inf #initialize best expected improvement (NOTE: maximizing expected_improvement = minimizing -expected_improvement)
		while too_close and not(success) and n_trials<=maxtrials:
			n_trials +=1
			for ii in range(maxtrials):
				x0 = np.asarray(xlow) + np.asarray(xup-xlow) * np.asarray(np.random.rand(1,dim)) #random starting point for optimizing expected improvement
				res= minimize(ei,x0,method='SLSQP',bounds=bound_list, args=(gpr_obj, Xsamples, Ysamples))
				if res.success == False:
					continue

				dist = np.min(scp.distance.cdist(np.asmatrix(res.x), Xsamples)) #make sure new point is sufficiently far away from already sampled points
				if np.min(dist)>1e-3 and res.success: #1e-3 is tunable
					if len(xnew) ==0:
						xnew = np.asmatrix(res.x)
						too_close = False
						success=True
						fnew = res.fun
					else:
						#if selecting only one point, do:
						if res.fun<fnew:
							fnew = res.fun
							xnew = np.asmatrix(res.x)
				else:
					x_ = np.asarray(xlow) + np.asarray(xup-xlow) * np.asarray(np.random.rand(1,dim)) #random starting point
					xnew = np.asmatrix(x_)
		print('xnew:',xnew)
		#evaluate objective at new point
		energy = objective(np.ravel(xnew))#obj_wrapper(np.ravel(xnew),objective, histfilename)
		print('energy', energy)
		#update Xsamples and Ysamples arrays
		Xsamples=np.concatenate((Xsamples, np.asmatrix(xnew)), axis = 0)
		Ysamples=np.concatenate((Ysamples, np.asmatrix(energy)), axis = 0)

		minID=np.argmin(Ysamples) #find index of best point
		print('bestX: ', Xsamples[minID,:])
		print('bestY: ', Ysamples[minID,:])
		print('Nevals: ', Xsamples.shape[0])

	#select start points for local search by balancing attained objective function value and distance to already selected start points
	opt_x = Xsamples[minID,:] #best parameters found so far
	opt_y = Ysamples[minID,:] #best ennergy found so far

	yall_sc = (Ysamples- min(Ysamples))/(max(Ysamples) -min(Ysamples)) #scale objective values to [0,1], small number better
	startp = opt_x #first start point is best solution returned by GP
	starty =opt_y
	NUM_DATAPOINTS=5 #set number of points from which local search will start - tunnable
	weights=np.linspace(0,1,NUM_DATAPOINTS) #weights to trade off: starting from best point vs from point furthest away and in between
	for jj in range(NUM_DATAPOINTS-1):
		d = np.min(scp.distance.cdist(Xsamples, startp), axis = 1).reshape(-1,1)
		d_sc = (max(d) - d)/(max(d)-min(d)) #scale distances to [0,1], large distances are better
		sum_sc = weights[jj+1]*d_sc +(1-weights[jj+1])*yall_sc
		minID=np.argmin(sum_sc)
		xadd = Xsamples[minID,:]
		startp=np.concatenate((startp, xadd), axis = 0)
		yadd = Ysamples[minID,:]
		starty=np.concatenate((starty, yadd), axis = 0)
	print('start points for ImFil', startp)
	starts=np.concatenate((startp, starty), axis = 1)
	budget = maxevals - Xsamples.shape[0] #remaining budget after GP is done

	#begin local search as in main file
	if budget > 0:
		#extra_opts = { 'scale_step' : 2,'scale_start' : 1,  'scale_depth' : 8, 'extra_argument' : 1, 'extra_arg_value': (objective, histfilename)}
		optimizer = IMFIL
		optimizer = optimizer(maxiter=budget)
		for ii in range(NUM_DATAPOINTS):
			print('initial guess: ', startp[ii,:])
			result = optimizer.optimize(
				num_vars = objective.npar(),
				objective_function=objective,
				initial_point = np.ravel(startp[ii,:]),
				variable_bounds = bounds,
			 )

			#print(result[2], len(result[2]))
			'''
			f= open(histfilename,"a+") #save new input-output pair to file
			for jj in range(result[2].shape[0]):
				for kk in range(dim+1):
					f.write("%f  " % (result[2][jj,kk]))
				f.write("\n" )
			f.close()
			'''
			budget = budget - result[2]#.shape[0]
			print('number evals remaining', budget)
			print('ImFil trial number: ', ii,', result: ', result)
			print("estimated energy: %.5f" % result[1])  #.optimal_value)
			print("parameters:      ", result[0])        #.optimal_point)
			if result[1]<opt_y:
				print('found improvement!')
				opt_y = result[1]
				opt_x = result[0]
			print('best energy so far: ', opt_y)
			print('best parameters so far: ', opt_x)
			optimizer = IMFIL
			optimizer = optimizer(budget) #update optimizer with reduced budget
			if budget<=0:
				break
		print(result )
		r = list(result)
		r[2] = min(maxevals, maxevals-budget)
		result=tuple(r)
		print(result)
	else:
		result = (np.ravel(opt_x), np.ravel(opt_y), min(maxevals, maxevals-budget))

	#make plots - load file with sample points and energy values
	data = np.loadtxt(histfilename)
	XY =data[:,0:dim] #param 1, 2,...,dim
	Z= data[:,-1] #energy
	if dim == 2: #2-d plots of surface
		kernel = RBF()+WhiteKernel(noise_level=0.05)
		gpr = GaussianProcessRegressor(kernel=kernel, random_state=0,normalize_y=True, n_restarts_optimizer=10).fit(XY, Z)
		N=100 #generate 100 x 100 points on [-1,1] where GP predicts energy
		u = np.linspace(xlow[0], xup[0], N)
		v = np.linspace(xlow[1], xup[1], N)
		XX, YY = np.meshgrid(u, v)
		XX_long = np.asmatrix(np.reshape(XX,-1)).T
		YY_long = np.asmatrix(np.reshape(YY,-1)).T
		predX=np.concatenate((XX_long, YY_long),axis=1 )
		mu, sigma = gpr.predict(predX, return_std=True) #predict energy and uncertainty of energy prediction

		#make contour plot
		fig = plt.figure(figsize=(16, 12))
		ax1 = fig.add_subplot(121)
		cs = ax1.contourf(XX, YY, mu.reshape(100,100), 20)
		ax1.scatter(XY[0:n_GP_samples,0], XY[0:n_GP_samples,1],marker = 's',color='m', label='GP sample points')
		ax1.scatter(XY[n_GP_samples:,0], XY[n_GP_samples:,1],marker = 'x', color='g', label='ImFil sample points')
		ax1.scatter(np.ravel(startp[:,0]), np.ravel(startp[:,1]),marker = 'o', color='y', label='ImFil start points')
		ax1.scatter(np.ravel(opt_x)[0], np.ravel(opt_x)[1],color='r', label='best points')
		#ax1.scatter(starts[:,0], starts[:,1],  marker='s', color ='r', label='ImFil start')
		ax1.set_xlim(xlow[0], xup[0])
		ax1.set_ylim(xlow[1], xup[1])
		ax1.set(xlabel='Parameter 1', ylabel='Parameter 2')
		ax1.legend(loc=2)

		#make surface plot
		ax2 = fig.add_subplot(122,projection='3d')
		ax2.plot_surface(XX, YY, mu.reshape(100,100), rstride=1, cstride=1, cmap='viridis', edgecolor='none')
		ax2.set_xlim(xlow[0], xup[0])
		ax2.set_ylim(xlow[1], xup[1])
		ax2.set_xlabel('Parameter 1')
		ax2.set_ylabel('Parameter 2')
		ax2.set_zlabel('Energy')
		plt.savefig('samples.png')
		plt.close("all")

	#make progress plot
	sortZ=np.zeros(len(Z))
	sortZ[0]=Z[0]
	for ii in range(len(Z)-1):
		if sortZ[ii]<Z[ii+1]:
			sortZ[ii+1] = sortZ[ii]
		else:
			sortZ[ii+1]=Z[ii+1]

	fig = plt.figure(figsize=(16, 12))
	plt.plot(np.arange(1, len(Z)+1), sortZ)
	plt.xlabel('Number of function evaluations')
	plt.ylabel('Best function value found so far')
	plt.savefig("progress.png")

	return (np.ravel(opt_x), np.ravel(opt_y), maxevals-budget)


def ei(x, gpr_obj, Xsamples, Ysamples): #expected improvement
	dim = len(x)
	x= x.reshape(1, -1)

	min_dist=np.min(scp.distance.cdist(x, Xsamples))
	if min_dist<1e-3:
		expected_improvement=0.0
		return expected_improvement

	mu, sigma = gpr_obj.predict(x.reshape(1, -1), return_std=True)

	mu_sample = gpr_obj.predict(Xsamples)

	# Needed for noise-based model, otherwise use np.min(Ysamples).
	mu_sample_opt = np.min(mu_sample)

	#loss_optimum = np.min(Ysamples)
	# In case sigma equals zero
	with np.errstate(divide='ignore'):
		Z = (mu_sample_opt-mu) / sigma
		expected_improvement = (mu_sample_opt-mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
		expected_improvement[sigma == 0.0] == 0.0
	answer=-1.*expected_improvement[0,0] #to maximize EI, you minimize the negative of it
	return answer
