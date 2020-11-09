#ifndef	_PREPROCESS_H_
#define	_PREPROCESS_H_



int CaseMatch(char* pbSeq, UINT4 uDataLen);
int Filtering(char* pbOutput, const char* pbInput, UINT4 uDataLen);
int RC_Conversion(char* pbOutput, const char* pbInput, UINT4 uDataLen);


int Mask(char* pbOutput, const char* pbInput, UINT4 uStart, UINT4 uEnd);
int GetBaseCnt(const char* pbSeq, int nSeqLen, int* pnBaseCnt);
float GetScore(int* pnBaseCnt, int nLen);


#endif
