#ifndef	_MINER_H_
#define	_MINER_H_


/* 3 stage function */
int Preprocessing(char* pbQry, char* pbRC, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int SeedingAndExt(char* pbQry, char* pbRC, char* pbData, UINT4 uQryLen, UINT4 uDataLen);
int Seeding(char* pbQry, char* pbRC, char* pbData, UINT4 uQryLen, UINT4 uDataLen);



/* other function */
int GetParam();
UINT4 DataFileWrite1();
UINT4 DataFileWrite2();
int GetData(char* pbQry, UINT4 uQryLen, UINT4 mode);
int StrTrimLeft(char* pbSeq);
int StrTrimRight(char* pbSeq);
int InitGreedyMem();
int CleanGreedyMem();


#endif
