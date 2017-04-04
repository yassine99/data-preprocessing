import numpy as np

class sparseMatrix():
	def __init__(self,file=None,feature_number=None):
		self.file=file
		self.feature_number=feature_number
		self._check_params()
		self.reachedLastLine=False
		self.file_iterator=open(file,'r')

	def _check_params(self):
        	if self.file is None:
            		raise ValueError("a file  should be specified")
        	if type(self.file) !=str:
            		raise ValueError("file should be a file name")
        	if self.feature_number is None:
            		raise ValueError("feature_number should be specified")
        	if type(self.feature_number)!=int:
            		raise ValueError("feature_number should be an int")

	def nextChunk(self,chunk_size=25000):
    		if self.reachedLastLine:
    			return np.zeros((0,0)),np.zeros(0)
    		x=np.zeros((chunk_size,self.feature_number))
    		y=np.zeros(chunk_size)
    		kk=0
    		while kk<chunk_size:
            		l=self.file_iterator.readline()
            		if l=="":
                		self.reachedLastLine=True 
                		return x[0:kk,:],y[0:kk]
            		vector=l.replace("\n","").split(" ")
            		y[kk]=float(vector[0])
            		for l in vector[1:]:
            			a,v=l.split(':')
            			x[kk,int(a)]=float(v)
            		kk+=1
        	return x,y

