import numpy as np
import DQNalign.tool.util.function as function

class record_align():
    def __init__(self):
        self.Qtemp = ""
        self.Gtemp = ""
        self.Stemp = ""
        self.xtemp = []
        self.ytemp = []
        self.stemp = 0
        self.gtemp = 0

    def reverse(self, sizeS1, sizeS2):
        self.Qtemp = self.Qtemp[::-1]
        self.Gtemp = self.Gtemp[::-1]
        self.Stemp = self.Stemp[::-1]
        self.xtemp = list(np.subtract([sizeS1]*len(self.xtemp),self.xtemp[::-1]))
        self.ytemp = list(np.subtract([sizeS2]*len(self.ytemp),self.ytemp[::-1]))

    def shift(self, index, xshift, yshift):
        xtemp2 = self.xtemp[index:]
        ytemp2 = self.ytemp[index:]
        self.xtemp[index:] = list(np.add(xtemp2,[xshift]*len(xtemp2)))
        self.ytemp[index:] = list(np.add(ytemp2,[yshift]*len(ytemp2)))

    def record(self, x, y, a, s1, s2):
        score = 0

        if a == -1:
            if len(s1)==len(s2):
                s1 = s1.upper().replace(".","-")
                s2 = s2.upper().replace(".","-")
                self.Qtemp += s1
                same = (np.array(list(s1)) == np.array(list(s2)))
                gap1 = (np.array(list(s1)) == np.array(["-"]*len(s1)))
                gap2 = (np.array(list(s2)) == np.array(["-"]*len(s2)))
                tempmatch = ["|"]*len(s1)
                tempgap1 = ["-"]*len(s1)
                tempgap2 = ["+"]*len(s1)
                tempmismatch = [" "]*len(s1)
                score = np.sum(same)
                self.gtemp += np.sum(gap1) + np.sum(gap2)
                Gtemp = np.where(same,tempmatch,tempmismatch)
                Gtemp = np.where(gap1,tempgap1,Gtemp)
                self.Gtemp += "".join(list(np.where(gap2,tempgap2,Gtemp)))
                self.Stemp += s2
                onetemp = [1]*len(s1)
                zerotemp = [0]*len(s1)
                xtemp = list(range(x[0],x[0]+len(s1)))
                ytemp = list(range(y[0],y[0]+len(s2)))
                gtemp1 = np.where(gap1,onetemp,zerotemp)
                gtemp2 = np.where(gap2,onetemp,zerotemp)
                gtemp1 = np.cumsum(gtemp1)
                gtemp2 = np.cumsum(gtemp2)
                #print(np.subtract(xtemp,gtemp1))
                #print(np.subtract(ytemp,gtemp2))
                self.xtemp += list(np.subtract(xtemp,gtemp1))
                self.ytemp += list(np.subtract(ytemp,gtemp2))
            else:
                self.Qtemp += s1
                self.Stemp += s2
                longsize = max(len(s1),len(s2))
                longside = (len(s1)>len(s2))
                if longside:
                    self.Stemp += "-" * (len(s1)-len(s2))
                    self.gtemp += len(s1)-len(s2)
                    same = (np.array(list(s1[:len(s2)])) == np.array(list(s2)))
                    tempmatch = ["|"]*len(s2)
                    tempmismatch = [" "]*len(s2)
                    score = np.sum(same)
                    self.Gtemp += "".join(list(np.where(same,tempmatch,tempmismatch)))
                    self.Gtemp += "+" * (len(s1)-len(s2))
                    self.xtemp += list(range(x[0],x[1]))
                    self.ytemp += list(range(y[0],y[1]))
                    self.ytemp += [y[1]-1] * (len(s1)-len(s2))
                else:
                    self.Qtemp += "-" * (len(s2)-len(s1))
                    self.gtemp += len(s2)-len(s1)
                    same = (np.array(list(s1)) == np.array(list(s2[:len(s1)])))
                    tempmatch = ["|"]*len(s1)
                    tempmismatch = [" "]*len(s1)
                    score = np.sum(same)
                    self.Gtemp += "".join(list(np.where(same,tempmatch,tempmismatch)))
                    self.Gtemp += "-" * (len(s2)-len(s1))
                    self.xtemp += list(range(x[0],x[1]))
                    self.ytemp += list(range(y[0],y[1]))
                    self.xtemp += [x[1]-1] * (len(s2)-len(s1))


        elif a == 0:
            self.Qtemp += s1
            if s1 == s2:
                self.Gtemp += "|"
                score += 1
            else:
                self.Gtemp += " "
            self.Stemp += s2
            self.xtemp.append(x)
            self.ytemp.append(y)
        elif a == 1:
            self.Qtemp += s1
            self.Gtemp += "+"
            self.Stemp += "-"
            self.gtemp += 1
            self.xtemp.append(x)
            self.ytemp.append(y)
        elif a == 2:
            self.Qtemp += "-"
            self.Gtemp += "-"
            self.Stemp += s2
            self.gtemp += 1
            self.xtemp.append(x)
            self.ytemp.append(y)

        self.stemp += score
        return score

    def postprocessing(self):
        self.Qtemp2 = self.Qtemp[:]
        self.Gtemp2 = self.Gtemp[:]
        self.Stemp2 = self.Stemp[:]
        self.xtemp2 = self.xtemp[:]
        self.ytemp2 = self.ytemp[:]
        self.Qtemp = ""
        self.Gtemp = ""
        self.Stemp = ""
        self.xtemp = []
        self.ytemp = []
        state = 0		# state - 0: go ahead, 1: insertion, 2: deletion, 3: insertion to deletion, 4: deletion to insertion
        plus = 0
        minus = 0

        for i in range(len(self.Gtemp2)):
            if (state==0) and (self.Gtemp2[i]=="+"):
                state = 1
                plus = 1
                minus = 0
            elif (state==0) and (self.Gtemp2[i]=="-"):
                state = 2
                plus = 0
                minus = 1
            elif state and (not ((self.Gtemp2[i]=="+")or(self.Gtemp2[i]=="-"))):
                if state == 3:
                    self.gtemp -= minus
                    self.Qtemp += self.Qtemp2[i-plus-minus:i-2*minus]
                    self.Stemp += self.Stemp2[i-plus-minus:i-2*minus]
                    self.Gtemp += self.Gtemp2[i-plus-minus:i-2*minus]
                    self.xtemp += self.xtemp2[i-plus-minus:i-2*minus]
                    self.ytemp += self.ytemp2[i-plus-minus:i-2*minus]
                    for j in range(minus):
                        self.Qtemp += self.Qtemp2[i-2*minus+j]
                        self.Stemp += self.Stemp2[i-minus+j]
                        if self.Qtemp2[i-2*minus+j] == self.Stemp2[i-minus+j]:
                            self.Gtemp += "|"
                            self.stemp += 1
                        else:
                            self.Gtemp += " "
                        self.xtemp.append(self.xtemp2[i-2*minus+j])
                        self.ytemp.append(self.ytemp2[i-minus+j])
                elif state == 4:
                    self.gtemp -= plus
                    self.Qtemp += self.Qtemp2[i-plus-minus:i-2*plus]
                    self.Stemp += self.Stemp2[i-plus-minus:i-2*plus]
                    self.Gtemp += self.Gtemp2[i-plus-minus:i-2*plus]
                    self.xtemp += self.xtemp2[i-plus-minus:i-2*plus]
                    self.ytemp += self.ytemp2[i-plus-minus:i-2*plus]
                    for j in range(plus):
                        self.Qtemp += self.Qtemp2[i-plus+j]
                        self.Stemp += self.Stemp2[i-2*plus+j]
                        if self.Qtemp2[i-plus+j] == self.Stemp2[i-2*plus+j]:
                            self.Gtemp += "|"
                            self.stemp += 1
                        else:
                            self.Gtemp += " "
                        self.xtemp.append(self.xtemp2[i-plus+j])
                        self.ytemp.append(self.ytemp2[i-2*plus+j])
                elif state == 1:
                    self.Qtemp += self.Qtemp2[i-plus:i]
                    self.Stemp += self.Stemp2[i-plus:i]
                    self.Gtemp += self.Gtemp2[i-plus:i]
                    self.xtemp += self.xtemp2[i-plus:i]
                    self.ytemp += self.ytemp2[i-plus:i]
                elif state == 2:
                    self.Qtemp += self.Qtemp2[i-minus:i]
                    self.Stemp += self.Stemp2[i-minus:i]
                    self.Gtemp += self.Gtemp2[i-minus:i]
                    self.xtemp += self.xtemp2[i-minus:i]
                    self.ytemp += self.ytemp2[i-minus:i]

                self.Qtemp += self.Qtemp2[i]
                self.Stemp += self.Stemp2[i]
                self.Gtemp += self.Gtemp2[i]
                self.xtemp.append(self.xtemp2[i])
                self.ytemp.append(self.ytemp2[i])
                state = 0
                plus = 0
                minus = 0
            elif (state==1) and (self.Gtemp2[i]=="+"):
                plus += 1
            elif (state==2) and (self.Gtemp2[i]=="-"):
                minus += 1
            elif (state==1) and (self.Gtemp2[i]=="-"):
                state = 3
                minus += 1
            elif (state==2) and (self.Gtemp2[i]=="+"):
                state = 4
                plus += 1
            elif (state==3) and (self.Gtemp2[i]=="+"):
                self.gtemp -= minus
                self.Qtemp += self.Qtemp2[i-plus-minus:i-2*minus]
                self.Stemp += self.Stemp2[i-plus-minus:i-2*minus]
                self.Gtemp += self.Gtemp2[i-plus-minus:i-2*minus]
                self.xtemp += self.xtemp2[i-plus-minus:i-2*minus]
                self.ytemp += self.ytemp2[i-plus-minus:i-2*minus]
                for j in range(minus):
                    self.Qtemp += self.Qtemp2[i-2*minus+j]
                    self.Stemp += self.Stemp2[i-minus+j]
                    if self.Qtemp2[i-2*minus+j] == self.Stemp2[i-minus+j]:
                        self.Gtemp += "|"
                        self.stemp += 1
                    else:
                        self.Gtemp += " "
                    self.xtemp.append(self.xtemp2[i-2*minus+j])
                    self.ytemp.append(self.ytemp2[i-minus+j])
                state = 1
                plus = 1
                minus = 0
            elif (state==4) and (self.Gtemp2[i]=="-"):
                self.gtemp -= plus
                self.Qtemp += self.Qtemp2[i-plus-minus:i-2*plus]
                self.Stemp += self.Stemp2[i-plus-minus:i-2*plus]
                self.Gtemp += self.Gtemp2[i-plus-minus:i-2*plus]
                self.xtemp += self.xtemp2[i-plus-minus:i-2*plus]
                self.ytemp += self.ytemp2[i-plus-minus:i-2*plus]
                for j in range(plus):
                    self.Qtemp += self.Qtemp2[i-plus+j]
                    self.Stemp += self.Stemp2[i-2*plus+j]
                    if self.Qtemp2[i-plus+j] == self.Stemp2[i-2*plus+j]:
                        self.Gtemp += "|"
                        self.stemp += 1
                    else:
                        self.Gtemp += " "
                    self.xtemp.append(self.xtemp2[i-plus+j])
                    self.ytemp.append(self.ytemp2[i-2*plus+j])
                state = 2
                plus = 0
                minus = 1
            elif (state==3) and (self.Gtemp2[i]=="-"):
                minus += 1
                if minus == plus:
                    self.gtemp -= plus
                    for j in range(plus):
                        self.Qtemp += self.Qtemp2[i-2*plus+j+1]
                        self.Stemp += self.Stemp2[i-plus+j+1]
                        if self.Qtemp2[i-2*plus+j+1] == self.Stemp2[i-plus+j+1]:
                            self.Gtemp += "|"
                            self.stemp += 1
                        else:
                            self.Gtemp += " "
                        self.xtemp.append(self.xtemp2[i-2*plus+j+1])
                        self.ytemp.append(self.ytemp2[i-plus+j+1])
                    state = 0
                    plus = 0
                    minus = 0
            elif (state==4) and (self.Gtemp2[i]=="+"):
                plus += 1
                if plus == minus:
                    self.gtemp -= minus
                    for j in range(minus):
                        self.Qtemp += self.Qtemp2[i-minus+j+1]
                        self.Stemp += self.Stemp2[i-2*minus+j+1]
                        if self.Qtemp2[i-minus+j+1] == self.Stemp2[i-2*minus+j+1]:
                            self.Gtemp += "|"
                            self.stemp += 1
                        else:
                            self.Gtemp += " "
                        self.xtemp.append(self.xtemp2[i-minus+j+1])
                        self.ytemp.append(self.ytemp2[i-2*minus+j+1])
                    state = 0
                    plus = 0
                    minus = 0
            else:
                self.Qtemp += self.Qtemp2[i]
                self.Stemp += self.Stemp2[i]
                self.Gtemp += self.Gtemp2[i]
                self.xtemp.append(self.xtemp2[i])
                self.ytemp.append(self.ytemp2[i])
            

    def print(self, file):
        self.postprocessing()

        file.write("Alignment results:\n")
        file.write("Exact matches: "+str(self.stemp)+"\n")

        file.write("Gaps: "+str(self.gtemp)+"\n"+"\n")

        for i in range(int(np.ceil(len(self.Qtemp)/60))):
            file.write('{:<20}'.format("Query "+str(self.xtemp[60*i])))
            file.write(self.Qtemp[60*i:60*i+60]+"\n")
            file.write('{:<20}'.format(""))
            file.write(self.Gtemp[60*i:60*i+60]+"\n")
            file.write('{:<20}'.format("Sbjct "+str(self.ytemp[60*i])))
            file.write(self.Stemp[60*i:60*i+60]+"\n")
            file.write("\n")

class record_path():
    def __init__(self):
        self.path = []
        self.seq1 = []
        self.seq2 = []
        self.rev2 = []
        self.align1 = []
        self.align2 = []
        self.aligng = []
        self.rT1 = []
        self.rT2 = []
        self.Escore = []
        self.Bitscore = []
        self.index = []
        self.reward = [1,-1,-1]
        self.len1 = 0
        self.len2 = 0
        self.num = 0

    def set(self, seq1, seq2, path):
        self.seq1 = seq1
        self.seq2 = seq2
        self.rev2 = function.get_reverse(seq2)
        self.path = path
        self.len1 = len(seq1)
        self.len2 = len(seq2)
        self.num = len(path)

    def postprocessing(self):
        for i in range(self.num):
            print(i, "/", self.num,"post-processing")
            pathx = self.path[i][0][:]
            pathy = self.path[i][1][:]

            state = 0		# state - 0: go ahead, 1: insertion, 2: deletion, 3: insertion to deletion, 4: deletion to insertion
            plus = 0
            minus = 0

            self.path[i][0] = []
            self.path[i][1] = []

            lastx = -1
            lasty = -1

            for j in range(len(pathx)):
                if state== 0:
                    if pathx[j] == lastx:
                        state = 2
                        plus = 0
                        minus = 1
                    elif pathy[j] == lasty:
                        state = 1
                        plus = 1
                        minus = 0
                    else:
                        self.path[i][0].append(pathx[j])
                        self.path[i][1].append(pathy[j])
                        state = 0
                        plus = 0
                        minus = 0
                elif state == 1:
                    if pathx[j] == lastx:
                        state = 3
                        minus += 1
                    elif pathy[j] == lasty:
                        plus += 1
                    else:
                        self.path[i][0] += pathx[j-plus:j+1]
                        self.path[i][1] += pathy[j-plus:j+1]
                        state = 0
                        plus = 0
                        minus = 0
                elif state == 2:
                    if pathx[j] == lastx:
                        minus += 1
                    elif pathy[j] == lasty:
                        state = 4
                        plus += 1
                    else:
                        self.path[i][0] += pathx[j-minus:j+1]
                        self.path[i][1] += pathy[j-minus:j+1]
                        state = 0
                        plus = 0
                        minus = 0
                elif state == 3:
                    if pathx[j] == lastx:
                        minus += 1
                        if plus == minus:
                            self.path[i][0] += pathx[j-plus-minus+1:j-minus+1]
                            self.path[i][1] += pathy[j-minus+1:j+1]
                            state = 0
                            plus = 0
                            minus = 0
                    elif pathy[j] == lasty:
                        self.path[i][0] += pathx[j-plus-minus:j-2*minus]
                        self.path[i][1] += pathy[j-plus-minus:j-2*minus]
                        self.path[i][0] += pathx[j-2*minus:j-minus]
                        self.path[i][1] += pathy[j-minus:j]
                        state = 1
                        plus = 1
                        minus = 0
                    else:
                        self.path[i][0] += pathx[j-plus-minus:j-2*minus]
                        self.path[i][1] += pathy[j-plus-minus:j-2*minus]
                        self.path[i][0] += pathx[j-2*minus:j-minus]
                        self.path[i][1] += pathy[j-minus:j]
                        self.path[i][0].append(pathx[j])
                        self.path[i][1].append(pathy[j])
                        state = 0
                        plus = 0
                        minus = 0
                elif state == 4:
                    if pathx[j] == lastx:
                        self.path[i][0] += pathx[j-plus-minus:j-2*plus]
                        self.path[i][1] += pathy[j-plus-minus:j-2*plus]
                        self.path[i][0] += pathx[j-plus:j]
                        self.path[i][1] += pathy[j-2*plus:j-plus]
                        state = 2
                        plus = 0
                        minus = 1
                    elif pathy[j] == lasty:
                        plus += 1
                        if plus == minus:
                            self.path[i][0] += pathx[j-plus+1:j+1]
                            self.path[i][1] += pathy[j-plus-minus+1:j-plus+1]
                            state = 0
                            plus = 0
                            minus = 0
                    else:
                        self.path[i][0] += pathx[j-plus-minus:j-2*plus]
                        self.path[i][1] += pathy[j-plus-minus:j-2*plus]
                        self.path[i][0] += pathx[j-plus:j]
                        self.path[i][1] += pathy[j-2*plus:j-plus]
                        self.path[i][0].append(pathx[j])
                        self.path[i][1].append(pathy[j])
                        state = 0
                        plus = 0
                        minus = 0

                lastx = pathx[j]
                lasty = pathy[j]

    def rescoring(self):
        self.rT1 = []
        self.rT2 = []
        self.align1 = []
        self.align2 = []
        for i in range(self.num):
            print(i, "/", self.num,"rescoring")
            if self.path[i][1][0] < self.path[i][1][-1]:
                pathx = [-1]+self.path[i][0][:]
                pathy = [-1]+self.path[i][1][:]
                gapx = np.where(np.array(pathx[1:])==np.array(pathx[:-1]))
                gapy = np.where(np.array(pathy[1:])==np.array(pathy[:-1]))
                tempx = np.array(list(self.seq1))
                tempy = np.array(list(self.seq2))
                strx = tempx[self.path[i][0]]
                stry = tempy[self.path[i][1]]

                strg = "|"*np.size(self.path[i][0])
                strg = np.array(list(strg))
            
                strx[gapx] = "-"
                stry[gapy] = "-"

                diff = np.where(strx!=stry)

                strg[diff] = " "
                strg[gapx] = "-"
                strg[gapy] = "+"

                self.align1.append(str(strx.tostring().decode("utf-16").replace("\x00","")))
                self.align2.append(str(stry.tostring().decode("utf-16").replace("\x00","")))
                self.aligng.append(str(strg.tostring().decode("utf-16").replace("\x00","")))

                rT1 = self.reward[0]*np.sum(strx==stry)+self.reward[1]*(np.sum(strx!=stry)-np.size(gapx)-np.size(gapx))+self.reward[2]*np.size(gapx)+self.reward[2]*np.size(gapy)
                rT2 = np.sum(strx==stry)

                self.rT1.append(rT1)
                self.rT2.append(rT2)
            else:
                pathx = [-1]+self.path[i][0][:]
                pathy = [-1]+self.path[i][1][:]
                gapx = np.where(np.array(pathx[1:])==np.array(pathx[:-1]))
                gapy = np.where(np.array(pathy[1:])==np.array(pathy[:-1]))
                tempx = np.array(list(self.seq1))
                tempy = np.array(list(self.rev2))
                strx = tempx[self.path[i][0]]
                stry = tempy[self.path[i][1]]

                strg = "|"*np.size(self.path[i][0])
                strg = np.array(list(strg))
            
                strx[gapx] = "-"
                stry[gapy] = "-"

                diff = np.where(strx!=stry)

                strg[diff] = " "
                strg[gapx] = "-"
                strg[gapy] = "+"

                self.align1.append(str(strx.tostring().decode("utf-16").replace("\x00","")))
                self.align2.append(str(stry.tostring().decode("utf-16").replace("\x00","")))
                self.aligng.append(str(strg.tostring().decode("utf-16").replace("\x00","")))

                rT1 = self.reward[0]*np.sum(strx==stry)+self.reward[1]*(np.sum(strx!=stry)-np.size(gapx)-np.size(gapx))+self.reward[2]*np.size(gapx)+self.reward[2]*np.size(gapy)
                rT2 = np.sum(strx==stry)

                self.rT1.append(rT1)
                self.rT2.append(rT2)

    def sortbyEscore(self):
        K = 0.0005
        Lambda = 0.1451
        self.Escore = []
        self.index = []

        for i in range(self.num):
            self.Escore.append(K*np.abs(self.path[i][0][-1]-self.path[i][0][0])*np.abs(self.path[i][1][-1]-self.path[i][1][0])*np.exp(-Lambda * self.rT1[i]))

        #print(self.Escore)
        self.index = list(np.argsort(np.array(self.Escore)))

    def sortbyBitscore(self):
        K = 0.0005
        Lambda = 0.1451
        self.Bitscore = []
        self.index = []

        for i in range(self.num):
            self.Bitscore.append((Lambda*self.rT1[i]-np.log(K))/np.log(2))

        #print(self.Bitscore)
        self.index = list(np.argsort(np.array(self.Bitscore)))[::-1]


    def print(self, file):
        self.postprocessing()
        self.rescoring()
        #self.sortbyEscore()
        self.sortbyBitscore()

        for i in range(self.num):
            print(i, "/", self.num, "print alignment")
            file.write(str(i)+"th alignment\n")
            file.write("score : "+str(self.rT1[self.index[i]])+"\n")
            #file.write("E-value : "+str(self.Escore[self.index[i]])+"\n")
            file.write("Bit-score : "+str(self.Bitscore[self.index[i]])+"\n")
            file.write("exact match : "+str(self.rT2[self.index[i]])+"\n")
            file.write("seq1 : "+str(self.path[self.index[i]][0][0])+"-"+str(self.path[self.index[i]][0][-1])+"\n")
            file.write("seq2 : "+str(self.path[self.index[i]][1][0])+"-"+str(self.path[self.index[i]][1][-1])+"\n")
            for j in range(int(np.ceil(np.size(self.path[self.index[i]][0])/60))):
                file.write("\n")
                file.write('{:<20}'.format("Query "+str(self.path[self.index[i]][0][60*j])))
                file.write(self.align1[self.index[i]][60*j:60*j+60])
                file.write("\n")
                file.write('{:<20}'.format(""))
                file.write(self.aligng[self.index[i]][60*j:60*j+60])
                file.write("\n")
                file.write('{:<20}'.format("Sbjct "+str(self.path[self.index[i]][1][60*j])))
                file.write(self.align2[self.index[i]][60*j:60*j+60])
                file.write("\n")

            file.write("\n")
