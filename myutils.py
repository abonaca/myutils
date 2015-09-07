# General utilities
from math import *
import inspect
import numpy as np
import sys
import astropy
from astropy.table import Table

# arrays
def extractm(x, M):
	"""Return every Mth element of x (array-like)"""
	
	N = np.size(x)
	inc = np.arange(N)
	ind = np.where(inc%M==0)
	
	return x[ind]

def printcol(*arg, **kwarg):
	"""Print vectors in columns
	Use: printcol <vec1> <vec2> .. <vecn> (<fout='path to file'>)
	Default: fout=sys.stdout"""

	# Set output
	if kwarg:
		f=open(kwarg['fout'],'w')
	else:
		f=sys.stdout
	
	# Get variable names
	frame = inspect.currentframe()
	frame2 = inspect.getouterframes(frame)[1]
	string = inspect.getframeinfo(frame2[0]).code_context[0].strip()
	args = string[string.find('(') + 1:-1].split(',')

	names = []
	for i in args:
		if i.find('=') != -1:
			names.append(i.split('=')[1].strip())
		else:
			names.append(i)
	
	Ncol=len(arg)
	Nrow=np.zeros(Ncol)
	
	for i in range(Ncol):
		Nrow[i]=len(arg[i])
	
	Nmax=int(np.max(Nrow))
	
	# Print
	print>>f,("#"),
	for i in range(len(names)):
		print>>f,("%12s\t"%names[i]),
	print>>f,("\n#\n"),
		
	for i in range(Nmax):
		for j in range(Ncol):
			if i<Nrow[j]:
				print>>f,('%12g\t'%arg[j][i]),
			else:
				print>>f,('\t'),
		print>>f,('\n'),
		
	if kwarg:
		f.close()


# manipulate astropy tables
def extract_column(t, names):
	"""Return a list of columns from a table
	Parameters:
	t - table
	names - column names to extract
	Returns:
	list of columns"""
	
	lst=[]
	for i in names:
		lst.append(np.array(t[i]))
	return lst

def add_npcolumn(t, vec=None, name="", dtype='float', index=None):
	"""Add numpy array as a table column
	Parameters:
	t - astropy table
	vec - array to be added to the table, if None, adds an empty array (default: None)
	name - column name
	dtype - column type (default: float)
	index - order index for the array in the output table (default: None)
	Returns:
	vec - array added to the table"""
	
	if index==None:
		index = len(t.columns)
	
	if vec==None:
		vec = np.array(np.size(t))
		
	tvec = astropy.table.Column(vec, name=name, dtype=dtype)
	t.add_column(tvec, index=index)
	
	return vec


# binning
def bincen(bined):
	"""Returns bin centers from an input array of bin edge"""
	N=np.size(bined)-1
	binc=np.zeros(N)
	
	for i in range(N):
		binc[i]=(bined[i]+bined[i+1])/2.
	
	return binc

def in2val(index, delta, initial):
	return initial+index/delta
	
def val2in(value, delta, initial):
	return np.int64((value-initial)/delta)


# equatorial <-> cartesian
def d2r(deg):
	return deg*np.pi/180.

def r2d(rad):
	return rad*180./np.pi

def eq2car(ra, dec):
	"""Convert equatorial coordinates to cartesian
	Assumes ra, dec in radians"""
	
	x=np.cos(dec)*np.cos(ra)
	y=np.cos(dec)*np.sin(ra)
	z=np.sin(dec)
	
	return(x,y,z)

def car2eq(x,y,z):
	"""Convert cartesian coordinates to equatoria;
	Returns ra, dec in radians"""
	
	N=np.size(z)
	ra=np.zeros(N)
	for i in range(N):
		if(x[i]!=0):
			ra[i]=atan(y[i]/x[i])
		else:
			ra[i]=np.pi/2.
	dec=np.arcsin(z)
	
	return(ra,dec)


# Math
def points2line(p1, p2):
	"""Returns coefficients of a line passing through points p1 and p2
	Parameters: 
	p1 - tuple (x1, y1)
	p2 - tuple (x2, y2)"""
	
	a=(p2[1]-p1[1])/np.float(p2[0]-p1[0])
	b=p1[1]-p1[0]*a
	
	return [a,b]

def crossprodmat(a):
	"""Returns a cross product matrix [a]_x
	Assumes 3D"""
	A=np.matrix([0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0])
	A.shape = (3,3)
	return A

# Numerical recipes
def gauleg(x1, x2, x, w, n):
	eps=3.0e-11
	m=int((n+1)/2.)
	xm=0.5*(x2+x1)
	xl=0.5*(x2-x1)
	
	for i in range(1,m+1):
		z=cos(np.pi*(i-0.25)/(n+0.5))
		
		condition = True
		while condition:
			p1=1.0
			p2=0.0
			for j in range(1,n+1):
				p3=p2
				p2=p1
				p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j
			pp=n*(z*p1-p2)/(z*z-1.0)
			z1=z
			z=z1-p1/pp
			condition = abs(z-z1)>eps
		
		x[i-1]=xm-xl*z
		x[n-i]=xm+xl*z
		w[i-1]=2.0*xl/((1.0-z*z)*pp*pp)
		w[n-i]=w[i-1]

def callgauleg():
	n=20
	x=np.zeros(n)
	w=np.zeros(n)
	a=0
	b=np.pi/2.
	gauleg(a, b, x, w, n)
	
	fx=np.sin(x)
	print sum(fx*w)


	