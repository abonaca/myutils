# General utilities
from __future__ import print_function, division
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

def wherein(x, y):
    """Returns indices of x which correspond to elements of y"""
    
    xsorted = np.argsort(x)
    ypos = np.searchsorted(x[xsorted], y)
    indices = xsorted[ypos]
    
    return indices

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

def add_npcolumn(t, vec=np.empty(0), name="", dtype='float', index=None):
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
	
	if np.size(vec)==0:
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
	"""Convert cartesian coordinates to equatorial
	Returns ra, dec in radians"""
	
	N=np.size(z)
	ra=np.zeros(N)
	for i in range(N):
		if(x[i]!=0):
			ra[i]=np.arctan2(y[i], x[i])
		else:
			ra[i]=np.pi/2.
	dec=np.arcsin(z)
	
	return(ra,dec)


# interpolation
def between_lines(x, y, x1, y1, x2, y2):
    """check if points x,y are between lines defined with x1,y1 and x2,y2"""
    
    if y1[0]>y1[-1]:
        y1 = y1[::-1]
        x1 = x1[::-1]
        
    if y2[0]>y2[-1]:
        y2 = y2[::-1]
        x2 = x2[::-1]
    
    xin1 = np.interp(y,y1,x1)
    xin2 = np.interp(y,y2,x2)
    
    indin = (x>=xin1) & (x<=xin2)
    
    return indin


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

def rotmatrix(theta, i):
    """Returns 3x3 rotation matrix around axis i for angle theta (in deg)"""
    theta = np.radians(theta)
    cth = np.cos(theta)
    sth = np.sin(theta)
    
    sign = (-1)**i
    R2 = np.array([[cth, -sign*sth], [sign*sth, cth]])
    
    R = np.zeros((3,3))
    R[i][i] = 1
    
    if i==0:
        R[1:,1:] = R2
    elif i==1:
        R[0][0] = R2[0][0]
        R[0][2] = R2[0][1]
        R[2][0] = R2[1][0]
        R[2][2] = R2[1][1]
    elif i==2:
        R[:2,:2] = R2
    
    return R

def sph2cart(ra, dec):
    """Convert two angles on a unit sphere to a 3d vector"""
    
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    
    return (x, y, z)

def cart2sph(x, y, z):
    """Convert a 3d vector on a unit sphere to two angles"""
    
    ra = np.arctan2(y, x)
    dec = np.arcsin(z)
    
    ra[ra<0] += 2*np.pi
    
    return (ra, dec)

def rotate_angles(a, d, R):
    """Return angles a, d rotated by a 3x3 matrix R
    All angles are in degrees"""
    
    x_, y_, z_ = sph2cart(np.radians(a), np.radians(d))
    X = np.column_stack((x_, y_, z_))
    
    X_rot = np.zeros(np.shape(X))
    for i in range(np.size(x_)):
        X_rot[i] = np.dot(R, X[i])
    
    a_rot, d_rot = cart2sph(X_rot[:, 0], X_rot[:, 1], X_rot[:, 2])
    
    return (np.degrees(a_rot), np.degrees(d_rot))

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
	print(sum(fx*w))


	
