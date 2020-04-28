import numpy as np

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
            self.Qtemp += s1
            for i in range(len(s1)):
                if s1[i] == s2[i]:
                    self.Gtemp += "|"
                    score += 1
                else:
                    self.Gtemp += " "
            self.Stemp += s2
            self.xtemp += list(range(x[0],x[1]))
            self.ytemp += list(range(y[0],y[1]))
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

    def print(self, file):

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
