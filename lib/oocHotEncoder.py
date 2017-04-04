na_values=['', '#N/A', '#N/A', 'N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN','-nan',
'1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan']

class ooche():
	def __init__(self,columns=None,target=None,Id=None):
		self.columns=columns
		self.target=target
		self.Id=Id
		self._check_params()
		self.oldToNew={}
		self.nextInd=0
		self._alreadyAdded=[]

	def _check_params(self):
		if self.columns is None:
			raise ValueError("At least specify one column to encode")
		if self.target is None:
			raise ValueError("Specify target name")
		if self.Id is None:
			raise ValueError("Specify Id name")
		if not isinstance(self.columns, list):
			self.columns=[self.columns]

	def hotEncode(self,oldFile,newFile,sep=','):
		of=open(oldFile,'r')
		nf=open(newFile,'w')
		header=of.readline()
		cols=header.replace('\n','').split(sep)

		cc=self.nextInd
		thisTarget=-1
		thisId=-1
		for c in range(len(cols)):
			if cols[c]==self.target:
				thisTarget=c
				continue
			if cols[c]==self.Id:
				thisId=c
				continue
			if cols[c] in self._alreadyAdded:
				continue
			else:
				self._alreadyAdded.append(cols[c])
			if cols[c] not in self.columns:
				self.oldToNew[cols[c]]=cc
				cc+=1
				self.nextInd+=1
			else:
				self.oldToNew[cols[c]]={}
		
		IdList=[]
		for line in of:
			line=line.replace('\n','').split(sep)
			newLine=""
			if thisTarget==-1:
				newLine="1 "
			for i in range(len(line)):
				if line[i] in na_values:
					continue
				if i==thisTarget:
					newLine=line[i]+' '+newLine
					continue
				
				if i==thisId:
					IdList.append(line[i])
					continue
				
				new_ind=self.oldToNew[cols[i]]
				if type(new_ind)==dict:
					new_ind=self.oldToNew[cols[i]].get(line[i],self.nextInd)
					if new_ind==self.nextInd:
						self.oldToNew[cols[i]][line[i]]=self.nextInd
						self.nextInd+=1
					
					newLine+=str(new_ind)+":1 "
				else:
					newLine+=str(new_ind)+":"+line[i]+" "

			nf.write(newLine[:-1]+'\n')
		nf.close()
		of.close()
		return IdList



