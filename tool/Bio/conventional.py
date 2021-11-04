import sys
import numpy as np
import DQNalign.tool.util.ReadSeq as readseq
import cv2
import time
import os
np.set_printoptions(threshold=sys.maxsize)

class Clustal():
    # WE REFER https://github.com/etetoolkit/ext_apps/blob/master/src/clustal-omega-1.2.1/src/clustal/ktuple_pair.c
    # We implemented a pairwise alignment in the CLUSTAL Omega for adjusting the parameters

    def __init__(self, env, seq1, seq2, name = ['seq_1','seq_2']):
        # k-tuple based pairwise alignment is used in Clustal
        #self.K = env.K
        #self.signif = env.signif
        #self.window = env.window
        #self.wind_gap = env.wind_gap

        #self.K = 7
        #self.signif = 500
        #self.window = 5
        #self.wind_gap = 5

        #self.K = 10
        #self.signif = 100
        #self.window = 5
        #self.wind_gap = 5

        #self.K = 7
        #self.signif = 50
        #self.window = 50
        #self.wind_gap = 5

        self.K = 2
        self.signif = 4
        self.window = 4
        self.wind_gap = 5

        #self.K = 13
        #self.signif = 50
        #self.window = 0
        #self.wind_gap = 100

        self.kind = 5
        self.encode(seq1,seq2)
        self.name = [name[0],name[1]]
        #self.encode("AACGTTAACGTT","ACCGTTAACCGTT")

    def encode(self, seq1, seq2):
        if np.size(seq1)>0 and np.size(seq2)>0:
            if (seq1[0] == 0) or (seq1[0] == 1) or (seq1[0] == 2) or (seq1[0] == 3):
                seqTemp1 = seq1+1
                seqTemp2 = seq2+1
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
            else:
                seqTemp1 = np.zeros(len(seq1), dtype=int)
                for _ in range(len(seqTemp1)):
                    seqTemp1[_] = (seq1[_] == 'A') + 2 * (seq1[_] == 'C') + 3 * (seq1[_] == 'G') + 4 * (
                            seq1[_] == 'T')
                seqTemp2 = np.zeros(len(seq2), dtype=int)
                for _ in range(len(seqTemp2)):
                    seqTemp2[_] = (seq2[_] == 'A') + 2 * (seq2[_] == 'C') + 3 * (seq2[_] == 'G') + 4 * (
                            seq2[_] == 'T')
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
        else:
            self.seq1 = seq1
            self.seq2 = seq2
        self.x = 0
        self.y = 0
        self.sizeS1 = np.size(self.seq1)
        self.sizeS2 = np.size(self.seq2)

    def pair_align(self):
        start = time.time()
        now = time.time()
        self.make_ptrs()
        print("K-tuple lookup table generation is completed :",time.time()-now,time.time()-start)
        now = time.time()
        self.diag_score()
        print("Diagonal score are calculated",time.time()-now,time.time()-start)
        now = time.time()
        self.des_quick_sort()
        print("Quick sort for diagonal score is completed",time.time()-now,time.time()-start)
        now = time.time()
        self.flag_top_diag()
        print("The top diagonal components are flagged",time.time()-now,time.time()-start)
        now = time.time()
        self.connect()
        print("Pairwise alignment is completed",time.time()-now,time.time()-start)
        return np.max(self.accum[0,:])

    def align_process(self):
        start = time.time()
        now = time.time()
        self.make_ptrs()
        print("K-tuple lookup table generation is completed :",time.time()-now,time.time()-start)
        now = time.time()
        self.diag_score()
        print("Diagonal score are calculated",time.time()-now,time.time()-start)
        now = time.time()
        self.des_quick_sort()
        print("Quick sort for diagonal score is completed",time.time()-now,time.time()-start)
        now = time.time()
        self.flag_top_diag()
        print("The top diagonal components are flagged",time.time()-now,time.time()-start)
        now = time.time()
        self.connect()
        print("Pairwise alignment is completed",time.time()-now,time.time()-start)
        return self.accum

    def preprocess(self):
        start = time.time()
        now = time.time()
        self.make_ptrs()
        print("K-tuple lookup table generation is completed :",time.time()-now,time.time()-start)
        now = time.time()
        self.diag_score()
        print("Diagonal score are calculated",time.time()-now,time.time()-start)
        now = time.time()
        self.des_quick_sort()
        print("Quick sort for diagonal score is completed",time.time()-now,time.time()-start)
        now = time.time()
        self.flag_top_diag()
        print("The top diagonal components are flagged",time.time()-now,time.time()-start)
        now = time.time()
        self.record_anchor()
        print("Preprocessing procedure is completed",time.time()-now,time.time()-start)
        return self.anchors, self.anchore

    def make_ptrs(self):
        base = np.power(self.kind, range(self.K))
        self.ktup1 = np.zeros(self.sizeS1-self.K+1,dtype=int)
        self.ktup2 = np.zeros(self.sizeS2-self.K+1,dtype=int)
        self.max_aln_len = max(2*max(self.sizeS1, self.sizeS2),np.power(self.kind,self.K))
        self.pl1 = -np.ones(self.max_aln_len,dtype=int)
        self.pl2 = -np.ones(self.max_aln_len,dtype=int)
        self.tptr1 = -np.ones(self.max_aln_len,dtype=int)
        self.tptr2 = -np.ones(self.max_aln_len,dtype=int)

        for i in range(self.K):
            temp1 = np.multiply(self.seq1[i:self.sizeS1-self.K+i+1], base[i])
            temp2 = np.multiply(self.seq2[i:self.sizeS2-self.K+i+1], base[i])
            self.ktup1 = np.add(self.ktup1, temp1)
            self.ktup2 = np.add(self.ktup2, temp2)

        for i in range(self.sizeS1-self.K+1):
            self.tptr1[i] = self.pl1[self.ktup1[i]]
            self.pl1[self.ktup1[i]] = i

        for i in range(self.sizeS2-self.K+1):
            self.tptr2[i] = self.pl2[self.ktup2[i]]
            self.pl2[self.ktup2[i]] = i

    def diag_score(self):
        self.displ = np.zeros(self.sizeS1+self.sizeS2,dtype=int)
        limit = np.power(self.kind, self.K)

        for i in range(limit):
            if i % np.power(self.kind,self.K-2) == 0:
                now = time.time()
            vn1 = self.pl1[i]
            while True:
                if vn1 == -1:
                    break
                vn2 = self.pl2[i]
                while not (vn2 == -1):
                    osptr = vn1-vn2+self.sizeS2
                    self.displ[osptr] += 1
                    vn2 = self.tptr2[vn2]
                vn1 = self.tptr1[vn1]
            if i % np.power(self.kind,self.K-2) == np.power(self.kind,self.K-2)-1:
                i = i
                print("Scoring diagonal is completed :",i+1,"/",limit,"with",time.time()-now)

    def des_quick_sort(self):
        self.index = list(range(self.sizeS1+self.sizeS2))
        lst = np.zeros(50,dtype=int)
        ust = np.zeros(50,dtype=int)
        lst[0] = 0
        ust[0] = len(self.index)-1
        p = 0

        while (p >= 0):
            if (lst[p] >= ust[p]):
                p -= 1
            else:
                i = lst[p] - 1
                j = ust[p]
                pivlin = self.displ[j]
                while (i < j):
                    i =  i + 1
                    while (self.displ[i] < pivlin):
                        i = i + 1
                    j = j - 1
                    while (j > i):
                        if (self.displ[j] <= pivlin):
                            break
                        j = j - 1
                    if (i < j):
                        temp = self.displ[i]
                        self.displ[i] = self.displ[j]
                        self.displ[j] = temp

                        temp = self.index[i]
                        self.index[i] = self.index[j]
                        self.index[j] = temp

                j = ust[p]

                temp = self.displ[i]
                self.displ[i] = self.displ[j]
                self.displ[j] = temp

                temp = self.index[i]
                self.index[i] = self.index[j]
                self.index[j] = temp

                if (i - lst[p] < ust[p] - i):
                    lst[p+1] = lst[p]
                    ust[p+1] = i-1
                    lst[p] = i+1
                else:
                    lst[p+1] = i+1
                    ust[p+1] = ust[p]
                    ust[p] = i-1
                p = p + 1

    def flag_top_diag(self):
        self.slopes = np.zeros(self.sizeS1+self.sizeS2,dtype=int)
        j = self.sizeS1 + self.sizeS2 - self.signif
        if (j < 0):
            j = 0

        for i in range(self.sizeS1+self.sizeS2-1,j-1,-1):
            if (self.displ[i] > 0):
                pos = self.index[i]
                l = max(0,pos-self.window)
                m = min(self.sizeS1+self.sizeS2-1,pos+self.window)
                self.slopes[l:m+1] = 1

    def record_anchor(self):
        self.anchor = []
        self.anchors = []
        self.anchore = []

        for i in range(self.sizeS1-self.K+1):
            encrypt = self.ktup1[i]
            vn2 = self.pl2[encrypt]
            while True:
                if (vn2==-1):
                    break
                osptr = i - vn2 + self.sizeS2
                if (self.slopes[osptr]):
                    self.anchor.append([i,vn2])

                vn2 = self.tptr2[vn2]

        self.anchors.append(self.anchor[0])
        for i in range(len(self.anchor)-1):
            if self.anchor[i+1][0] - self.anchor[i][0] > self.wind_gap:
                self.anchore.append(self.anchor[i])
                self.anchors.append(self.anchor[i+1])

        self.anchore.append(self.anchor[-1])

    def connect(self):
        self.accum = np.zeros((5,2*self.max_aln_len),dtype=int)
        self.displ = np.zeros(self.sizeS1+self.sizeS2,dtype=int)
        curr_frag = 0
        self.maxsf = 0

        for i in range(self.sizeS1-self.K+1):
            if i % 10000 == 0:
                now = time.time()
            encrypt = self.ktup1[i]
            vn2 = self.pl2[encrypt]
            while True:
                if (vn2==-1):
                    flag = True
                    break
                osptr = i - vn2 + self.sizeS2
                if (not self.slopes[osptr]):
                    vn2 = self.tptr2[vn2]
                    continue
                flen = 0
                fs = self.K
                self.next = self.maxsf

                while True:
                    if (not self.next):
                        curr_frag += 1
                        if (curr_frag >= 2*self.max_aln_len-1):
                            os._exit(1)
                        self.displ[osptr] = curr_frag
                        self.put_frag(fs, i, vn2, flen, curr_frag)
                    else:
                        tv1 = self.accum[1][self.next]
                        tv2 = self.accum[2][self.next]

                        if (self.frag_rel_pos(i,vn2,tv1,tv2)):
                            if (i-vn2 == self.accum[1][self.next]-self.accum[2][self.next]):
                                if (i > self.accum[1][self.next] + self.K -1):
                                    fs = self.accum[0][self.next] + self.K
                                else:
                                    rmndr = i - self.accum[1][self.next]
                                    fs = self.accum[0][self.next]+rmndr
                                flen = self.next
                                self.next =0
                                continue
                            else:
                                if (not self.displ[osptr]):
                                    subt1 = self.K
                                else:
                                    if (i>self.accum[1][self.displ[osptr]]+self.K-1):
                                        subt1 = self.accum[0][self.displ[osptr]]+self.K
                                    else:
                                        rmndr = i-self.accum[1][self.displ[osptr]]
                                        subt1 = self.accum[0][self.displ[osptr]] + rmndr
                                subt2 = self.accum[0][self.next] - self.wind_gap + self.K
                                if (subt2>subt1):
                                    flen = self.next
                                    fs = subt2
                                else:
                                    flen = self.displ[osptr]
                                    fs = subt1
                                self.next = 0
                                continue
                        else:
                            self.next = self.accum[4][self.next]
                            continue

                    break

                vn2 = self.tptr2[vn2]

            if i % 10000 == 10000-1:
                i = i
                print("Align process is completed :",i+1,"/",self.sizeS1-self.K+1,"with",time.time()-now)

    def put_frag(self, fs, i, vn2, flen, curr_frag):
        self.accum[0][curr_frag] = fs
        self.accum[1][curr_frag] = i
        self.accum[2][curr_frag] = vn2
        self.accum[3][curr_frag] = flen

        if (self.maxsf < 0):
            self.maxsf = 0
            self.accum[4][curr_frag]=0
            return

        if (fs > self.accum[0][self.maxsf]):
            self.accum[4][curr_frag]=self.maxsf
            self.maxsf=curr_frag
            return
        else:
            self.next = self.maxsf
            while True:
                end = self.next
                self.next = self.accum[4][self.next]
                if (fs >= self.accum[0][self.next]):
                    break
            self.accum[4][curr_frag]=self.next
            self.accum[4][end]=curr_frag

        return

    def frag_rel_pos(self, i, vn2, tv1, tv2):
        if (i-vn2 == tv1-tv2):
            if (tv1 < i):
                return True
        else:
            if (tv1+self.K-1<i) and (tv2+self.K-1<vn2):
                return True
        return False

    def display(self, filename):
        dot_plot = 255*np.ones((self.sizeS1, self.sizeS2))
        max_score = np.max(self.accum[0][:])
        index = np.argmax(self.accum[0][:])
        dot_plot[self.accum[1][index]][self.accum[2][index]] = 0
        while self.accum[0][index] > 0:
            last_point = [self.accum[1][index],self.accum[2][index]]
            index = self.accum[3][index]
            if self.accum[1][index]-last_point[0] == self.accum[2][index]-last_point[1]:
                for i in range(self.accum[1][index],last_point[0]):
                    dot_plot[i][i-self.accum[1][index]+self.accum[2][index]]=0
            else:
                for i in range(self.accum[1][index],last_point[0]):
                    for j in range(self.accum[2][index],last_point[1]):
                        dot_plot[i][j]=0

        cv2.imwrite(filename,dot_plot)

    def print(self, filename, printsize=60):
        file = open(filename,'w')
        file.write("DQNalign Project v1.0\n")
        file.write("Python implemented pairwise alignment algorithm of Clustal Omega\n")
        file.write("Sequence 1 : "+self.name[0]+", length : "+str(self.sizeS1)+"\n")
        file.write("Sequence 2 : "+self.name[1]+", length : "+str(self.sizeS2)+"\n")
        file.write("\n")

        file.write("Alignment results:\n")
        max_score = np.max(self.accum[0][:])
        index = np.argmax(self.accum[0][:])
        file.write("Exact matches: "+str(max_score)+"\n")

        path = []

        while self.accum[0][index] > 0:
            last_point = [self.accum[1][index],self.accum[2][index]]
            index = self.accum[3][index]
            if self.accum[1][index]-last_point[0] == self.accum[2][index]-last_point[1]:
                for i in range(last_point[0]-self.accum[1][index]):
                    path.append([last_point[0]-i-1,last_point[1]-i-1])
            else:
                path.append([self.accum[1][index],self.accum[2][index]])

        path = path[::-1]
        if (self.seq1[path[0][0]] != self.seq2[path[0][1]]) or (path[1][0] - path[0][0] > 1) or (path[1][1] - path[0][1] > 1):
            path = path[1:]
        str_to_print = ""
        Qtemp = ""
        Gtemp = ""
        Stemp = ""
        xtemp = []
        ytemp = []
        gtemp = 0
        ptemp = [0,0]
        ttemp = 0
        Nucleotide = ["N","A","C","G","T"]
        nucleotide = ["n","a","c","g","t"]

        for i in range(len(path)):
            #print(ptemp, path[i])
            if ((path[i][0] - ptemp[0] > 1) or (path[i][1] - ptemp[1] > 1)) and (i > 0):
                for j in range(path[i][0] - ptemp[0]):
                    Qtemp += nucleotide[self.seq1[ptemp[0]+j+1]]

                for j in range(path[i][1] - ptemp[1]):
                    Stemp += nucleotide[self.seq2[ptemp[1]+j+1]]

                longsize = max(path[i][0] - ptemp[0],path[i][1] - ptemp[1])
                longside = (path[i][0] - ptemp[0] > path[i][1] - ptemp[1])

                if longside:
                    Stemp += "-" * ((path[i][0] - ptemp[0]) - (path[i][1] - ptemp[1]))
                    gtemp += (path[i][0] - ptemp[0]) - (path[i][1] - ptemp[1])
                    Gtemp += "+" * longsize
                    xtemp += list(range(ptemp[0]+1,path[i][0]+1))
                    ytemp += list(range(ptemp[1]+1,path[i][1]+1))
                    ytemp += [path[i][1]]*((path[i][0] - ptemp[0]) - (path[i][1] - ptemp[1]))
                else:
                    Qtemp += "-" * ((path[i][1] - ptemp[1]) - (path[i][0] - ptemp[0]))
                    gtemp += (path[i][1] - ptemp[1]) - (path[i][0] - ptemp[0])
                    Gtemp += "-" * longsize
                    xtemp += list(range(ptemp[0]+1,path[i][0]+1))
                    xtemp += [path[i][1]]*((path[i][1] - ptemp[1]) - (path[i][0] - ptemp[0]))
                    ytemp += list(range(ptemp[1]+1,path[i][1]+1))
            else:
                Qtemp += Nucleotide[self.seq1[path[i][0]]]
                Stemp += Nucleotide[self.seq2[path[i][1]]]
                if Nucleotide[self.seq1[path[i][0]]] == Nucleotide[self.seq2[path[i][1]]]:
                    Gtemp += "|"
                else:
                    Gtemp += " "
                xtemp.append(path[i][0])
                ytemp.append(path[i][1])
                
            ptemp = path[i]

        for i in range(int(np.ceil(len(Qtemp)/60))):
            temp = '{:<20}'.format("Query "+str(xtemp[60*i]))
            str_to_print += temp + Qtemp[60*i:60*i+60] + "\n"
            temp = '{:<20}'.format("")
            str_to_print += temp + Gtemp[60*i:60*i+60] + "\n"
            temp = '{:<20}'.format("Sbjct "+str(ytemp[60*i]))
            str_to_print += temp + Stemp[60*i:60*i+60] + "\n"
            str_to_print += "\n"
	    
        file.write("Gaps: "+str(gtemp)+"\n"+"\n")
        file.write(str_to_print)

        file.close()

class MUMmer():
    # The Mummer 3.23 software must be installed with command "sudo apt-get install mummer"
    # We just used the mummer software with

    def __init__(self, env, seq1file, seq2file, name = ['seq_1','seq_2'], outputname='ref_qry'):
        #self.max_gap = env.max_gap
        #self.min_cluster = env.min_cluster

        self.max_gap = 90
        self.min_cluster = 20

        #self.max_gap = 100000
        #self.min_cluster = 10000

        self.seq1file = seq1file
        self.seq2file = seq2file
        self.outputname = outputname
        self.name = [name[0],name[1]]
        self.coords1 = []
        self.coords2 = []
        self.aligns1 = []
        self.aligns2 = []
        self.score = 0

    def align(self):
        os.system("nucmer --maxgap=%d --mincluster=%d --prefix=%s %s %s" % (self.max_gap, self.min_cluster, self.outputname, self.seq1file, self.seq2file))
        os.system("show-coords -r %s.delta > %s.coords" % (self.outputname, self.outputname))
        os.system("show-aligns %s.delta %s %s > %s.aligns" % (self.outputname, self.name[0], self.name[1], self.outputname))

    def read_aligns(self):
        self.aligns1 = []
        self.aligns2 = []
        self.score = 0
        state = 0 ## 0 : find BEGIN alignment, 1: after BEGIN alignment (find END alignment)
        tempalign1 = ""
        tempalign2 = ""
        tempscore = 0
        tempgap = 0

        file = open(self.outputname+".aligns",'r')
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        temp = file.readline()
        while temp:
            if state == 0:
                aligninfo = temp.split()
                if np.size(aligninfo)>1:
                    if aligninfo[1] == "BEGIN":
                        #print(aligninfo)
                        state = 1
                        tempalign1 = ""
                        tempalign2 = ""
                        tempscore = 0
                        tempgap = 0
                        temp = file.readline()
            else:
                aligninfo1 = file.readline().replace("\n","").split()
                aligninfo2 = file.readline().replace("\n","").split()
                temp = file.readline()
                gapinfo = temp.replace("\n","").split()
                #print(aligninfo1)
                #print(aligninfo2)
                #print(gapinfo)

                if aligninfo2[1] == "END":
                    state = 0
                    self.aligns1.append(tempalign1)
                    self.aligns2.append(tempalign2)
                    self.score += tempscore - tempgap
                    #print(gapinfo)
                    continue

                tempalign1 += aligninfo1[1]
                tempalign2 += aligninfo2[1]
                tempscore += len(aligninfo1[1])
                for g in gapinfo:
                    tempgap += len(g)
                

            temp = file.readline()
        
        
    def read_coords(self):
        self.coords1 = []
        self.coords2 = []
        estimatedscore = 0

        file = open(self.outputname+".coords",'r')
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        temp = file.readline().replace("\n","")
        while temp:
            coordinfo = temp.split()
            self.coords1.append([int(coordinfo[0]), int(coordinfo[1])])
            self.coords2.append([int(coordinfo[3]), int(coordinfo[4])])
            estimatedscore += max(int(coordinfo[6]),int(coordinfo[7])) * float(coordinfo[9]) / 100
            temp = file.readline().replace("\n","")

        file.close()
        print("Estimated exact match score of the mummer result is :",estimatedscore)
            

    def export_info(self):
        self.read_coords()
        self.read_aligns()

        return self.coords1, self.coords2, self.aligns1, self.aligns2, self.score

    def print(self, name):
        os.system("cp %s.aligns %s.aligns" % (self.outputname, name))
        os.system("cp %s.coords %s.coords" % (self.outputname, name))

class BLAST():
    # The BLAST function is not implemented in this version
    def __init__(self, env, seq1file, seq2file, name = ['seq_1','seq_2'], outputname='ref_qry'):
        # Parameters for BLAST algorithms are written in env
        self.X = 100

        self.seq1file = seq1file
        self.seq2file = seq2file
        self.outputname = outputname
        self.name = [name[0],name[1]]
        self.coords1 = []
        self.coords2 = []
        self.aligns1 = []
        self.aligns2 = []
        self.score = 0

    def encode(self, seq1, seq2):
        if np.size(seq1)>0 and np.size(seq2)>0:
            if (seq1[0] == 0) or (seq1[0] == 1) or (seq1[0] == 2) or (seq1[0] == 3) :
                seqTemp1 = seq1
                seqTemp2 = seq2
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
            else:
                seqTemp1 = np.zeros(len(seq1), dtype=int)
                for _ in range(len(seqTemp1)):
                    seqTemp1[_] = (seq1[_] == 'A') + 2 * (seq1[_] == 'C') + 3 * (seq1[_] == 'G') + 4 * (
                            seq1[_] == 'T') - 1
                seqTemp2 = np.zeros(len(seq2), dtype=int)
                for _ in range(len(seqTemp2)):
                    seqTemp2[_] = (seq2[_] == 'A') + 2 * (seq2[_] == 'C') + 3 * (seq2[_] == 'G') + 4 * (
                            seq2[_] == 'T') - 1
                self.seq1 = seqTemp1.astype(int)
                self.seq2 = seqTemp2.astype(int)
        else:
            self.seq1 = seq1
            self.seq2 = seq2
        self.x = 0
        self.y = 0
        self.sizeS1 = np.size(self.seq1)
        self.sizeS2 = np.size(self.seq2)

    def align(self):
        os.system("blastn -export_search_strategy blast_param_%d.txt -gapopen 2 -gapextend 2 -reward 1 -penalty -1 -query %s -subject %s -xdrop_ungap %d -xdrop_gap %d -xdrop_gap_final %d -out %s.out" % (self.X,  self.seq1file, self.seq2file, self.X, self.X, self.X, self.outputname))

    def print(self, name):
        os.system("cp %s.out %s.out" % (self.outputname, name))
