#ifndef	_SEED_H_
#define	_SEED_H_



int WordMatch(char* pbQryM, UINT4 uQryLen, PWORD_ELE pstWordEle, PTMP_WORD pstTmpWord);


int NHitMethodQry(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int NHitMethodQryAllow(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int NHitMethodRC(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int NHitMethodRCAllow(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);

int NHitMethodQry2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int NHitMethodQryAllow2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int NHitMethodRC2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int NHitMethodRCAllow2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);


UINT4 GetWord(char* pbSeq, int* pnNoBaseOffset);
UINT4 GetWord(char* pbSeq);
UINT4 GetRCWord(char* pbSeq);
UINT4 GetAllowSeqNum(UINT4 uAllowSize);
UINT4 GetAllowSeq(UINT4 uSeq, UINT4 uOrder);


int SeedMakeQry(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int SeedMakeRC(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int SeedFindQry(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int SeedFindRC(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int CleanTmpSeedListQry(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int CleanTmpSeedListRC(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int CleanTmpSeedListQry2(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int CleanTmpSeedListRC2(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int WordCombine(PTMP_SEED pstTmpSeed, int nSpace);
int SeedCombine(PTMP_SEED_LIST pstTmpSeedList, UINT4 uX, UINT4 uY, long long llDiag, char* pbSeq1, char* pbSeq2);
UINT4 ExtToThrForward(char* pbSeq1, char* pbSeq2, UINT4 uMaxLen, int* pnScore);
UINT4 ExtToThrBackward(char* pbSeq1, char* pbSeq2, UINT4 uMaxLen, int* pnScore);
int IsOnlyBase(char* pbSeq1, char* pbSeq2, UINT4 uSeqLen);
UINT4 GetSeedMatNum(char* pbSeq1, char* pbSeq2, UINT4 uSeqLen);


int GappedExtMgrQry(PREAL_SEED_LIST pstRealSeedList, PTMP_SEED pstTmpSeed, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int GappedExtMgrRC(PREAL_SEED_LIST pstRealSeedList, PTMP_SEED pstTmpSeed, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int CleanRealSeedListQry(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int CleanRealSeedListRC(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);
int CleanRealSeedListQry2(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int CleanRealSeedListRC2(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen);

int WriteSeedFileQry(PSEED pstSEED, UINT4 uQryLen, UINT4 uDataLen);
int WriteSeedFileRC(PSEED pstSEED, UINT4 uQryLen, UINT4 uDataLen);


#endif
