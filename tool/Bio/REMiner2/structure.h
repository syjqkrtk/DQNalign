#ifndef	_STRUCTURE_H_
#define	_STRUCTURE_H_



typedef struct tag_WordEle
{
	UINT4	uIndex;			// Initial value is infinity
	UINT4	uPrevIndex;		// Previous index of same word (infinity when nothing)

} WORD_ELE, *PWORD_ELE;

typedef struct tag_TmpWord
{
	UINT4	uIndex;			// Initial value is infinity

} TMP_WORD, *PTMP_WORD;

typedef struct tag_TmpSeed
{
	UINT4	uX;
	UINT4	uY;
	UINT4	uLen;
	UINT4	uMatNum;

} TMP_SEED, *PTMP_SEED;

typedef struct tag_TmpSeedList
{
	long long	llDiagNum;

	TMP_SEED*	pTmpSeed;

} TMP_SEED_LIST, *PTMP_SEED_LIST;

typedef struct tag_RealSeedList
{
	int			nSeedNum;

	TMP_SEED*	pTmpSeed;

} REAL_SEED_LIST, *PREAL_SEED_LIST;

typedef struct tag_RE
{
	UINT4	uX1;
	UINT4	uX2;
	UINT4	uY1;
	UINT4	uY2;
	float	fIdentity;
	UINT4	uSeedX;
	UINT4	uSeedY;
	UINT4	uSeedLen;

} RE, *PRE;

typedef struct tag_SEED
{
	UINT4	uX1;
	UINT4	uX2;
	UINT4	uY1;
	UINT4	uY2;

} SEED, *PSEED;



/* tmp seed list function */
int InitTmpSeedList(PTMP_SEED_LIST pstTmpSeedList, long long llDiagNum);	// initilize tmp seed list
int UpdateTmpSeed(PTMP_SEED_LIST pstTmpSeedList, long long llDiag, UINT4 uX, UINT4 uY, UINT4 uLen, UINT4 uMatNum);	// update a tmp seed
int UpdateTmpSeed(PTMP_SEED_LIST pstTmpSeedList, long long llDiag, UINT4 uX, UINT4 uY, UINT4 uLen);					// update a tmp seed
int FreeTmpSeedList(PTMP_SEED_LIST pstTmpSeedList);							// free seed list

/* real seed list function */
int InitRealSeedList(PREAL_SEED_LIST pstRealSeedList, int nSeedNum);		// initilize real seed list
int PushRealSeed(PREAL_SEED_LIST pstRealSeedList, PTMP_SEED pstTmpSeed);	// push a real seed
TMP_SEED GetRealSeed(PREAL_SEED_LIST pstRealSeedList, int nSeedIndex);		// get a real seed
int ResetRealSeedList(PREAL_SEED_LIST pstRealSeedList);						// reset real seed list
int FreeRealSeedList(PREAL_SEED_LIST pstRealSeedList);						// free real seed list



#endif
