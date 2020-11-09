#ifndef	_GVD_H_
#define	_GVD_H_

typedef unsigned int UINT4;


/* macro */
#define MAX2(A,B)		( ((A) > (B)) ? (A) : (B) )
#define MIN2(A,B)		( ((A) < (B)) ? (A) : (B) )


/* map */
const UINT4 auBase2NumMap[26] =
{
	0,	10,	1,	10,	10,
	10,	2,	10,	10,	10,
	10,	10,	10,	4,	10,
	5,	10,	10,	10,	3,
	10,	10,	10,	10,	10,	10		// for 26 alphabets (A:0, C:1, G:2, T:3, N:4, P:5, reminder:10)
};

const UINT4 auBase2CNumMap[26] =
{
	3,	10,	2,	10,	10,
	10,	1,	10,	10,	10,
	10,	10,	10,	4,	10,
	5,	10,	10,	10,	0,
	10,	10,	10,	10,	10,	10		// for 26 alphabets (A:3, C:2, G:1, T:0, N:4, P:5, reminder:10)
};

const char abMaskMap[26] =
{
	'a',	'B',	'c',	'D',	'E',
	'F',	'g',	'H',	'I',	'J',
	'K',	'L',	'M',	'N',	'O',
	'P',	'Q',	'R',	'S',	't',
	'U',	'V',	'W',	'X',	'Y',	'Z'		// for 26 alphabets (A->a, C->c, G->g, T->t, reminder:same)
};

const char abComplementMap[26] =
{
	'T',	'B',	'G',	'D',	'E',
	'F',	'C',	'H',	'I',	'J',
	'K',	'L',	'M',	'N',	'O',
	'P',	'Q',	'R',	'S',	'A',
	'U',	'V',	'W',	'X',	'Y',	'Z'		// for 26 alphabets (A<->T, C<->G, reminder:same)
};


/* file*/
#define QRY_FILE		"query.seq"		// input1 을 정리한 파일 (주석 & CR 정리, case matching)
#define QRY_M_FILE		"query_m.seq"	// input1 을 정리하고 filtering 한 파일
#define DATA_FILE		"data.seq"		// input2 을 정리한 파일 (주석 & CR 정리, case matching)
#define DATA_M_FILE		"data_m.seq"		// input2 을 정리한 파일 (주석 & CR 정리, case matching)
#define RC_FILE			"reverse.seq"	// input2 을 reverse complement 한 파일
#define RES_FILE		"result.rem"	// result 파일
#define SEED_FILE		"result.seed"	// result 파일
#define CSV_FILE		"result.csv"	// result 파일
#define LOG_FILE		"Log.txt"		// log 파일
#define PARAM_FILE		"param.txt"		// user parameter 파일


/* system define variable */
#define ASCII_A			65					// 'A'
#define ASCII_Z			90					// 'Z'
#define INFINITY4U		0xFFFFFFFF			// 4 bytes의 최대수 (unsigned)
#define INFINITY4		0x7FFFFFFF			// 4 bytes의 최대수 (signed)
#define INFINITY8U		0xFFFFFFFFFFFFFFFF	// 8 bytes의 최대수 (unsigned)
#define INFINITY8		0x7FFFFFFFFFFFFFFF	// 8 bytes의 최대수 (signed)
#define REAL_SEED_NUM	1000000				// gapped extension 전에 쌓아놓는 seed 개수 (A) (메모리사용: 16A) (최대 2G)
#define BLOCK_LEN		10240				// file을 한번에 읽는 단위
#define INPUT_LEN		256					// 최대 input file 이름 길이
#define PARAM			1					// user에게 파라미터를 받을지 결정


/* user define variable */
#if PARAM
extern	char	INPUT_FILE1[INPUT_LEN];
extern	char	INPUT_FILE2[INPUT_LEN];
extern	UINT4	WORD_SIZE;
extern	int		ALLOW_SIZE;
extern	UINT4	SPACE_SIZE;
extern	UINT4	MIN_SEED_LEN;
extern	int		SCORE_MAT;
extern	int		SCORE_MIS;
extern	int		SCORE_THR;
extern	int		GREEDY_X;
extern	int		GREEDY_MIN_L;
extern	int		GREEDY_MAX_U;
extern	int		WD_SIZE;
extern	float	T_THR;
extern	int		ALIGN_MODE;

#else
#define INPUT_FILE1		"Y.001.seq"		// input 1 파일 (488K)
#define INPUT_FILE2		"Y.002.seq"		// input 2 파일 (488K)
#define WORD_SIZE		12				// WORD 크기 (16이하로 해야)
#define ALLOW_SIZE		0				// word 당 allowable 크기 (0, 1만 가능)
#define SPACE_SIZE		2				// SPACE 크기
#define MIN_SEED_LEN	32				// seed를 이루기 위한 최소 seed 길이
#define SCORE_MAT		1				// match score (정수)
#define SCORE_MIS		-2				// mismatch score (정수)
#define SCORE_THR		-4				// score threshold
#define GREEDY_X		10				// greedy algorithm에서 X 값
#define GREEDY_MIN_L	-1024			// greedy algorithm에서 L의 최소값
#define GREEDY_MAX_U	1024			// greedy algorithm에서 U의 최대값
#define WD_SIZE			10				// filtering을 위한 window 값
#define T_THR			0.9				// filtering을 위한 threshold 값
#define ALIGN_MODE		1				// alignment mode
#endif


/* extern variable */
extern	FILE*		g_pfRes;		// 결과 파일

extern	long long**	g_ppllGreedyPreR;
extern	long long**	g_ppllGreedyCurR;
extern	UINT4**		g_ppuPreMatNumArr;
extern	UINT4**		g_ppuCurMatNumArr;



#endif
