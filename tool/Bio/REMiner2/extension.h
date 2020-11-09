#ifndef	_EXTENSION_H_
#define	_EXTENSION_H_



int UngappedExt(PTMP_SEED pstTmpSeed, char* pbSeq1, char* pbSeq2, UINT4 uQryLen, UINT4 uDataLen);

int GappedExtQry(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int GappedExtRC(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int GappedExtForward(char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen, PRE pstRE, UINT4* puMatNum, UINT4* puDiff, int nThreadNum);
int GappedExtBackward(char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen, PRE pstRE, UINT4* puMatNum, UINT4* puDiff, int nThreadNum);


float GetGreedySp(long long llI, long long llJ, long long llD);
long long GetGreedyMaxI(long long llGreedyK, long long llGreedyL, long long llGreedyU, long long* pllGreedyPreR, char* pbSeq1, char* pbSeq2, UINT4* puPreMatNumArr, UINT4* puCurMatNumArr);
long long GetGreedyMinI(long long llGreedyK, long long llGreedyL, long long llGreedyU, long long* pllGreedyPreR, char* pbSeq1, char* pbSeq2, UINT4 uX, UINT4 uY, UINT4* puPreMatNumArr, UINT4* puCurMatNumArr);


int WriteResFileQry(PRE pstRE, UINT4 uQryLen, UINT4 uDataLen);
int WriteResFileRC(PRE pstRE, UINT4 uQryLen, UINT4 uDataLen);



#endif
