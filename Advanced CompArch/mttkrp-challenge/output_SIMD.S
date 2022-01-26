
mttkrp:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000001190 <_init>:
    1190:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1194:	910003fd 	mov	x29, sp
    1198:	940002bf 	bl	1c94 <call_weak_fn>
    119c:	a8c17bfd 	ldp	x29, x30, [sp], #16
    11a0:	d65f03c0 	ret

Disassembly of section .plt:

00000000000011b0 <.plt>:
    11b0:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
    11b4:	b00000b0 	adrp	x16, 16000 <__FRAME_END__+0xf684>
    11b8:	f947fe11 	ldr	x17, [x16, #4088]
    11bc:	913fe210 	add	x16, x16, #0xff8
    11c0:	d61f0220 	br	x17
    11c4:	d503201f 	nop
    11c8:	d503201f 	nop
    11cc:	d503201f 	nop

00000000000011d0 <memcpy@plt>:
    11d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11d4:	f9400211 	ldr	x17, [x16]
    11d8:	91000210 	add	x16, x16, #0x0
    11dc:	d61f0220 	br	x17

00000000000011e0 <exit@plt>:
    11e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11e4:	f9400611 	ldr	x17, [x16, #8]
    11e8:	91002210 	add	x16, x16, #0x8
    11ec:	d61f0220 	br	x17

00000000000011f0 <__cxa_finalize@plt>:
    11f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11f4:	f9400a11 	ldr	x17, [x16, #16]
    11f8:	91004210 	add	x16, x16, #0x10
    11fc:	d61f0220 	br	x17

0000000000001200 <fputc@plt>:
    1200:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1204:	f9400e11 	ldr	x17, [x16, #24]
    1208:	91006210 	add	x16, x16, #0x18
    120c:	d61f0220 	br	x17

0000000000001210 <clock_gettime@plt>:
    1210:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1214:	f9401211 	ldr	x17, [x16, #32]
    1218:	91008210 	add	x16, x16, #0x20
    121c:	d61f0220 	br	x17

0000000000001220 <asprintf@plt>:
    1220:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1224:	f9401611 	ldr	x17, [x16, #40]
    1228:	9100a210 	add	x16, x16, #0x28
    122c:	d61f0220 	br	x17

0000000000001230 <pow@plt>:
    1230:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1234:	f9401a11 	ldr	x17, [x16, #48]
    1238:	9100c210 	add	x16, x16, #0x30
    123c:	d61f0220 	br	x17

0000000000001240 <omp_get_thread_num@plt>:
    1240:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1244:	f9401e11 	ldr	x17, [x16, #56]
    1248:	9100e210 	add	x16, x16, #0x38
    124c:	d61f0220 	br	x17

0000000000001250 <fclose@plt>:
    1250:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1254:	f9402211 	ldr	x17, [x16, #64]
    1258:	91010210 	add	x16, x16, #0x40
    125c:	d61f0220 	br	x17

0000000000001260 <fopen@plt>:
    1260:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1264:	f9402611 	ldr	x17, [x16, #72]
    1268:	91012210 	add	x16, x16, #0x48
    126c:	d61f0220 	br	x17

0000000000001270 <time@plt>:
    1270:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1274:	f9402a11 	ldr	x17, [x16, #80]
    1278:	91014210 	add	x16, x16, #0x50
    127c:	d61f0220 	br	x17

0000000000001280 <malloc@plt>:
    1280:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1284:	f9402e11 	ldr	x17, [x16, #88]
    1288:	91016210 	add	x16, x16, #0x58
    128c:	d61f0220 	br	x17

0000000000001290 <__isoc99_fscanf@plt>:
    1290:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1294:	f9403211 	ldr	x17, [x16, #96]
    1298:	91018210 	add	x16, x16, #0x60
    129c:	d61f0220 	br	x17

00000000000012a0 <__libc_start_main@plt>:
    12a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12a4:	f9403611 	ldr	x17, [x16, #104]
    12a8:	9101a210 	add	x16, x16, #0x68
    12ac:	d61f0220 	br	x17

00000000000012b0 <fgetc@plt>:
    12b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12b4:	f9403a11 	ldr	x17, [x16, #112]
    12b8:	9101c210 	add	x16, x16, #0x70
    12bc:	d61f0220 	br	x17

00000000000012c0 <aligned_alloc@plt>:
    12c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12c4:	f9403e11 	ldr	x17, [x16, #120]
    12c8:	9101e210 	add	x16, x16, #0x78
    12cc:	d61f0220 	br	x17

00000000000012d0 <memset@plt>:
    12d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12d4:	f9404211 	ldr	x17, [x16, #128]
    12d8:	91020210 	add	x16, x16, #0x80
    12dc:	d61f0220 	br	x17

00000000000012e0 <calloc@plt>:
    12e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12e4:	f9404611 	ldr	x17, [x16, #136]
    12e8:	91022210 	add	x16, x16, #0x88
    12ec:	d61f0220 	br	x17

00000000000012f0 <realloc@plt>:
    12f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12f4:	f9404a11 	ldr	x17, [x16, #144]
    12f8:	91024210 	add	x16, x16, #0x90
    12fc:	d61f0220 	br	x17

0000000000001300 <rand@plt>:
    1300:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1304:	f9404e11 	ldr	x17, [x16, #152]
    1308:	91026210 	add	x16, x16, #0x98
    130c:	d61f0220 	br	x17

0000000000001310 <strdup@plt>:
    1310:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1314:	f9405211 	ldr	x17, [x16, #160]
    1318:	91028210 	add	x16, x16, #0xa0
    131c:	d61f0220 	br	x17

0000000000001320 <omp_get_num_threads@plt>:
    1320:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1324:	f9405611 	ldr	x17, [x16, #168]
    1328:	9102a210 	add	x16, x16, #0xa8
    132c:	d61f0220 	br	x17

0000000000001330 <strerror@plt>:
    1330:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1334:	f9405a11 	ldr	x17, [x16, #176]
    1338:	9102c210 	add	x16, x16, #0xb0
    133c:	d61f0220 	br	x17

0000000000001340 <strrchr@plt>:
    1340:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1344:	f9405e11 	ldr	x17, [x16, #184]
    1348:	9102e210 	add	x16, x16, #0xb8
    134c:	d61f0220 	br	x17

0000000000001350 <__gmon_start__@plt>:
    1350:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1354:	f9406211 	ldr	x17, [x16, #192]
    1358:	91030210 	add	x16, x16, #0xc0
    135c:	d61f0220 	br	x17

0000000000001360 <__getauxval@plt>:
    1360:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1364:	f9406611 	ldr	x17, [x16, #200]
    1368:	91032210 	add	x16, x16, #0xc8
    136c:	d61f0220 	br	x17

0000000000001370 <abort@plt>:
    1370:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1374:	f9406a11 	ldr	x17, [x16, #208]
    1378:	91034210 	add	x16, x16, #0xd0
    137c:	d61f0220 	br	x17

0000000000001380 <puts@plt>:
    1380:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1384:	f9406e11 	ldr	x17, [x16, #216]
    1388:	91036210 	add	x16, x16, #0xd8
    138c:	d61f0220 	br	x17

0000000000001390 <getopt_long@plt>:
    1390:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1394:	f9407211 	ldr	x17, [x16, #224]
    1398:	91038210 	add	x16, x16, #0xe0
    139c:	d61f0220 	br	x17

00000000000013a0 <strcmp@plt>:
    13a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13a4:	f9407611 	ldr	x17, [x16, #232]
    13a8:	9103a210 	add	x16, x16, #0xe8
    13ac:	d61f0220 	br	x17

00000000000013b0 <fread@plt>:
    13b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13b4:	f9407a11 	ldr	x17, [x16, #240]
    13b8:	9103c210 	add	x16, x16, #0xf0
    13bc:	d61f0220 	br	x17

00000000000013c0 <free@plt>:
    13c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13c4:	f9407e11 	ldr	x17, [x16, #248]
    13c8:	9103e210 	add	x16, x16, #0xf8
    13cc:	d61f0220 	br	x17

00000000000013d0 <fwrite@plt>:
    13d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13d4:	f9408211 	ldr	x17, [x16, #256]
    13d8:	91040210 	add	x16, x16, #0x100
    13dc:	d61f0220 	br	x17

00000000000013e0 <fflush@plt>:
    13e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13e4:	f9408611 	ldr	x17, [x16, #264]
    13e8:	91042210 	add	x16, x16, #0x108
    13ec:	d61f0220 	br	x17

00000000000013f0 <strcpy@plt>:
    13f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13f4:	f9408a11 	ldr	x17, [x16, #272]
    13f8:	91044210 	add	x16, x16, #0x110
    13fc:	d61f0220 	br	x17

0000000000001400 <srand@plt>:
    1400:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1404:	f9408e11 	ldr	x17, [x16, #280]
    1408:	91046210 	add	x16, x16, #0x118
    140c:	d61f0220 	br	x17

0000000000001410 <__isoc99_sscanf@plt>:
    1410:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1414:	f9409211 	ldr	x17, [x16, #288]
    1418:	91048210 	add	x16, x16, #0x120
    141c:	d61f0220 	br	x17

0000000000001420 <printf@plt>:
    1420:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1424:	f9409611 	ldr	x17, [x16, #296]
    1428:	9104a210 	add	x16, x16, #0x128
    142c:	d61f0220 	br	x17

0000000000001430 <__errno_location@plt>:
    1430:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1434:	f9409a11 	ldr	x17, [x16, #304]
    1438:	9104c210 	add	x16, x16, #0x130
    143c:	d61f0220 	br	x17

0000000000001440 <GOMP_parallel@plt>:
    1440:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1444:	f9409e11 	ldr	x17, [x16, #312]
    1448:	9104e210 	add	x16, x16, #0x138
    144c:	d61f0220 	br	x17

0000000000001450 <putchar@plt>:
    1450:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1454:	f940a211 	ldr	x17, [x16, #320]
    1458:	91050210 	add	x16, x16, #0x140
    145c:	d61f0220 	br	x17

0000000000001460 <fprintf@plt>:
    1460:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1464:	f940a611 	ldr	x17, [x16, #328]
    1468:	91052210 	add	x16, x16, #0x148
    146c:	d61f0220 	br	x17

Disassembly of section .text:

0000000000001470 <main>:

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in COO format, matrices are dense.
 */
int main(int argc, char ** argv)
{
    1470:	d13203ff 	sub	sp, sp, #0xc80
	sptSparseTensor X;
	sptMatrix ** U;

	bool random = true;
	sptIndex mode = 0;
	sptIndex R = 16;
    1474:	52800203 	mov	w3, #0x10                  	// #16
	int dev_id = -2;
    1478:	12800022 	mov	w2, #0xfffffffe            	// #-2
	int niters = 5;
	int nthreads = 1;
	printf("niters: %d\n", niters);
    147c:	90000024 	adrp	x4, 5000 <sptBytesString+0x80>
{
    1480:	a9007bfd 	stp	x29, x30, [sp]
    1484:	910003fd 	mov	x29, sp
    1488:	a90153f3 	stp	x19, x20, [sp, #16]
    148c:	2a0003f3 	mov	w19, w0
    1490:	aa0103f4 	mov	x20, x1
	printf("niters: %d\n", niters);
    1494:	91100080 	add	x0, x4, #0x400
    1498:	528000a1 	mov	w1, #0x5                   	// #5
{
    149c:	a9025bf5 	stp	x21, x22, [sp, #32]
    14a0:	a90363f7 	stp	x23, x24, [sp, #48]
    14a4:	a9046bf9 	stp	x25, x26, [sp, #64]
    14a8:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex R = 16;
    14ac:	290e0fff 	stp	wzr, w3, [sp, #112]
	int dev_id = -2;
    14b0:	b9007be2 	str	w2, [sp, #120]
	printf("niters: %d\n", niters);
    14b4:	97ffffdb 	bl	1420 <printf@plt>

	if(argc <= 3) { // #Required arguments
    14b8:	71000e7f 	cmp	w19, #0x3
    14bc:	54003a6d 	b.le	1c08 <main+0x798>
			case 'i':
				strcpy(fname, optarg);
				printf("input file: %s\n", fname); fflush(stdout);
				break;
			case 'o':
				fo = fopen(optarg, "w");
    14c0:	b00000ba 	adrp	x26, 16000 <__FRAME_END__+0xf684>
    14c4:	d00000b7 	adrp	x23, 17000 <memcpy@GLIBC_2.17>
    14c8:	f947df5b 	ldr	x27, [x26, #4024]
    14cc:	9000003c 	adrp	x28, 5000 <sptBytesString+0x80>
    14d0:	912b839c 	add	x28, x28, #0xae0
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14d4:	910582f7 	add	x23, x23, #0x160
	bool random = true;
    14d8:	52800035 	mov	w21, #0x1                   	// #1
	FILE *fo = NULL;
    14dc:	d2800016 	mov	x22, #0x0                   	// #0
    14e0:	9101f3f9 	add	x25, sp, #0x7c
    14e4:	90000038 	adrp	x24, 5000 <sptBytesString+0x80>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14e8:	aa1903e4 	mov	x4, x25
    14ec:	aa1703e3 	mov	x3, x23
		int option_index = 0;
    14f0:	b9007fff 	str	wzr, [sp, #124]
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14f4:	91104302 	add	x2, x24, #0x410
    14f8:	aa1403e1 	mov	x1, x20
    14fc:	2a1303e0 	mov	w0, w19
    1500:	97ffffa4 	bl	1390 <getopt_long@plt>
		if(c == -1) {
    1504:	3100041f 	cmn	w0, #0x1
    1508:	54000ae0 	b.eq	1664 <main+0x1f4>  // b.none
		switch(c) {
    150c:	7101bc1f 	cmp	w0, #0x6f
    1510:	54000900 	b.eq	1630 <main+0x1c0>  // b.none
    1514:	5400018c 	b.gt	1544 <main+0xd4>
    1518:	7101a41f 	cmp	w0, #0x69
    151c:	54000760 	b.eq	1608 <main+0x198>  // b.none
    1520:	7101b41f 	cmp	w0, #0x6d
    1524:	54000221 	b.ne	1568 <main+0xf8>  // b.any
				strcpy(foname, optarg);
				sptAssert(fo != NULL);
				printf("output file: %s\n", optarg); fflush(stdout);
				break;
			case 'm':
				sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
    1528:	f947df40 	ldr	x0, [x26, #4024]
    152c:	90000021 	adrp	x1, 5000 <sptBytesString+0x80>
    1530:	9101c3e2 	add	x2, sp, #0x70
    1534:	91112021 	add	x1, x1, #0x448
    1538:	f9400000 	ldr	x0, [x0]
    153c:	97ffffb5 	bl	1410 <__isoc99_sscanf@plt>
				break;
    1540:	17ffffea 	b	14e8 <main+0x78>
		switch(c) {
    1544:	7101c81f 	cmp	w0, #0x72
    1548:	540003c1 	b.ne	15c0 <main+0x150>  // b.any
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
					exit(1);
				}
				break;
			case 'r':
				sscanf(optarg, "%u"PASTA_SCN_INDEX, &R);
    154c:	f947df40 	ldr	x0, [x26, #4024]
    1550:	90000021 	adrp	x1, 5000 <sptBytesString+0x80>
    1554:	9101d3e2 	add	x2, sp, #0x74
    1558:	9111e021 	add	x1, x1, #0x478
    155c:	f9400000 	ldr	x0, [x0]
    1560:	97ffffac 	bl	1410 <__isoc99_sscanf@plt>
				break;
    1564:	17ffffe1 	b	14e8 <main+0x78>
		switch(c) {
    1568:	7101901f 	cmp	w0, #0x64
    156c:	540034e1 	b.ne	1c08 <main+0x798>  // b.any
				sscanf(optarg, "%d", &dev_id);
    1570:	f947df40 	ldr	x0, [x26, #4024]
    1574:	90000021 	adrp	x1, 5000 <sptBytesString+0x80>
    1578:	9101e3e2 	add	x2, sp, #0x78
    157c:	91114021 	add	x1, x1, #0x450
    1580:	f9400000 	ldr	x0, [x0]
    1584:	97ffffa3 	bl	1410 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    1588:	b9407be0 	ldr	w0, [sp, #120]
    158c:	11000800 	add	w0, w0, #0x2
    1590:	7100041f 	cmp	w0, #0x1
    1594:	54fffaa9 	b.ls	14e8 <main+0x78>  // b.plast
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    1598:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xf684>
    159c:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    15a0:	f947d863 	ldr	x3, [x3, #4016]
    15a4:	91116000 	add	x0, x0, #0x458
    15a8:	d2800382 	mov	x2, #0x1c                  	// #28
    15ac:	d2800021 	mov	x1, #0x1                   	// #1
    15b0:	f9400063 	ldr	x3, [x3]
    15b4:	97ffff87 	bl	13d0 <fwrite@plt>
					exit(1);
    15b8:	52800020 	mov	w0, #0x1                   	// #1
    15bc:	97ffff09 	bl	11e0 <exit@plt>
		switch(c) {
    15c0:	7101d81f 	cmp	w0, #0x76
    15c4:	54003221 	b.ne	1c08 <main+0x798>  // b.any
			case 'v':
				random = false;
				strcpy(fvname, optarg);
    15c8:	f947df41 	ldr	x1, [x26, #4024]
    15cc:	9112c3e2 	add	x2, sp, #0x4b0
				random = false;
    15d0:	52800015 	mov	w21, #0x0                   	// #0
				strcpy(fvname, optarg);
    15d4:	aa0203e0 	mov	x0, x2
    15d8:	f9400021 	ldr	x1, [x1]
    15dc:	97ffff85 	bl	13f0 <strcpy@plt>
    15e0:	aa0003e2 	mov	x2, x0
				printf("validation input file: %s\n", fvname); fflush(stdout);
    15e4:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    15e8:	aa0203e1 	mov	x1, x2
    15ec:	91120000 	add	x0, x0, #0x480
				printf("output file: %s\n", optarg); fflush(stdout);
    15f0:	97ffff8c 	bl	1420 <printf@plt>
    15f4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    15f8:	f947e000 	ldr	x0, [x0, #4032]
    15fc:	f9400000 	ldr	x0, [x0]
    1600:	97ffff78 	bl	13e0 <fflush@plt>
				break;
    1604:	17ffffb9 	b	14e8 <main+0x78>
				strcpy(fname, optarg);
    1608:	f947df41 	ldr	x1, [x26, #4024]
    160c:	910323e2 	add	x2, sp, #0xc8
    1610:	aa0203e0 	mov	x0, x2
    1614:	f9400021 	ldr	x1, [x1]
    1618:	97ffff76 	bl	13f0 <strcpy@plt>
    161c:	aa0003e2 	mov	x2, x0
				printf("input file: %s\n", fname); fflush(stdout);
    1620:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1624:	aa0203e1 	mov	x1, x2
    1628:	91108000 	add	x0, x0, #0x420
    162c:	17fffff1 	b	15f0 <main+0x180>
				fo = fopen(optarg, "w");
    1630:	aa1c03e1 	mov	x1, x28
    1634:	f9400360 	ldr	x0, [x27]
    1638:	97ffff0a 	bl	1260 <fopen@plt>
				strcpy(foname, optarg);
    163c:	f9400361 	ldr	x1, [x27]
				fo = fopen(optarg, "w");
    1640:	aa0003f6 	mov	x22, x0
				strcpy(foname, optarg);
    1644:	912263e0 	add	x0, sp, #0x898
    1648:	f90037e1 	str	x1, [sp, #104]
    164c:	97ffff69 	bl	13f0 <strcpy@plt>
				sptAssert(fo != NULL);
    1650:	f94037e1 	ldr	x1, [sp, #104]
    1654:	b4002196 	cbz	x22, 1a84 <main+0x614>
				printf("output file: %s\n", optarg); fflush(stdout);
    1658:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    165c:	9110c000 	add	x0, x0, #0x430
    1660:	17ffffe4 	b	15f0 <main+0x180>
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1664:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1668:	b94073e1 	ldr	w1, [sp, #112]
	printf("dev_id: %d\n", dev_id);

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    166c:	910223f9 	add	x25, sp, #0x88
	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1670:	91128000 	add	x0, x0, #0x4a0
    1674:	97ffff6b 	bl	1420 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    1678:	b9407be1 	ldr	w1, [sp, #120]
    167c:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1680:	9112c000 	add	x0, x0, #0x4b0
    1684:	97ffff67 	bl	1420 <printf@plt>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1688:	aa1903e0 	mov	x0, x25
    168c:	910323e2 	add	x2, sp, #0xc8
    1690:	52800021 	mov	w1, #0x1                   	// #1
    1694:	94000473 	bl	2860 <sptLoadSparseTensor>
    1698:	35001f60 	cbnz	w0, 1a84 <main+0x614>
	sptSparseTensorStatus(&X, stdout);
    169c:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xf684>
    16a0:	aa1903e0 	mov	x0, x25
    16a4:	f947e021 	ldr	x1, [x1, #4032]
    16a8:	f9400021 	ldr	x1, [x1]
    16ac:	9400080d 	bl	36e0 <sptSparseTensorStatus>

	sptIndex nmodes = X.nmodes;
    16b0:	b9408bf3 	ldr	w19, [sp, #136]
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    16b4:	11000660 	add	w0, w19, #0x1
    16b8:	1100067a 	add	w26, w19, #0x1
    16bc:	d37df000 	lsl	x0, x0, #3
    16c0:	97fffef0 	bl	1280 <malloc@plt>
    16c4:	8b3a4c18 	add	x24, x0, w26, uxtw #3
    16c8:	aa0003f4 	mov	x20, x0
	for(sptIndex m=0; m<nmodes+1; ++m) {
    16cc:	aa0003f7 	mov	x23, x0
    16d0:	3400013a 	cbz	w26, 16f4 <main+0x284>
    16d4:	d503201f 	nop
		U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    16d8:	d2800300 	mov	x0, #0x18                  	// #24
    16dc:	97fffee9 	bl	1280 <malloc@plt>
    16e0:	f80086e0 	str	x0, [x23], #8
	for(sptIndex m=0; m<nmodes+1; ++m) {
    16e4:	eb1802ff 	cmp	x23, x24
    16e8:	54ffff81 	b.ne	16d8 <main+0x268>  // b.any
	}
	sptIndex max_ndims = 0;
    16ec:	52800018 	mov	w24, #0x0                   	// #0
	for(sptIndex m=0; m<nmodes; ++m) {
    16f0:	340002b3 	cbz	w19, 1744 <main+0x2d4>
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    16f4:	f9404fe0 	ldr	x0, [sp, #152]
    16f8:	d2800017 	mov	x23, #0x0                   	// #0
    16fc:	52800018 	mov	w24, #0x0                   	// #0
    1700:	d37ef6fa 	lsl	x26, x23, #2
    1704:	b8777801 	ldr	w1, [x0, x23, lsl #2]
    1708:	b94077e2 	ldr	w2, [sp, #116]
    170c:	f8777a80 	ldr	x0, [x20, x23, lsl #3]
    1710:	940006dc 	bl	3280 <sptNewMatrix>
    1714:	35001b80 	cbnz	w0, 1a84 <main+0x614>
		// sptAssert(sptConstantMatrix(U[m], 1) == 0);
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    1718:	2a1503e1 	mov	w1, w21
    171c:	f8777a80 	ldr	x0, [x20, x23, lsl #3]
    1720:	94000708 	bl	3340 <sptRandomizeMatrix>
    1724:	35001b00 	cbnz	w0, 1a84 <main+0x614>
		if(X.ndims[m] > max_ndims)
    1728:	f9404fe0 	ldr	x0, [sp, #152]
	for(sptIndex m=0; m<nmodes; ++m) {
    172c:	910006f7 	add	x23, x23, #0x1
    1730:	b87a6801 	ldr	w1, [x0, x26]
    1734:	6b01031f 	cmp	w24, w1
    1738:	1a812318 	csel	w24, w24, w1, cs  // cs = hs, nlast
    173c:	6b17027f 	cmp	w19, w23
    1740:	54fffe08 	b.hi	1700 <main+0x290>  // b.pmore
			max_ndims = X.ndims[m];
	}
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1744:	d37d7e60 	ubfiz	x0, x19, #3, #32
    1748:	2a1303fb 	mov	w27, w19
    174c:	b94077e2 	ldr	w2, [sp, #116]
    1750:	aa0003e3 	mov	x3, x0
    1754:	2a1803e1 	mov	w1, w24
    1758:	f87b7a80 	ldr	x0, [x20, x27, lsl #3]
    175c:	8b03029a 	add	x26, x20, x3
    1760:	f90037e3 	str	x3, [sp, #104]
    1764:	940006c7 	bl	3280 <sptNewMatrix>
    1768:	350018e0 	cbnz	w0, 1a84 <main+0x614>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    176c:	0f000400 	movi	v0.2s, #0x0
    1770:	f87b7a80 	ldr	x0, [x20, x27, lsl #3]
    1774:	94000737 	bl	3450 <sptConstantMatrix>
    1778:	35001860 	cbnz	w0, 1a84 <main+0x614>
	sptIndex stride = U[0]->stride;

	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    177c:	d37ef760 	lsl	x0, x27, #2
    1780:	97fffec0 	bl	1280 <malloc@plt>
	mats_order[0] = mode;
    1784:	b94073e3 	ldr	w3, [sp, #112]
	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    1788:	aa0003f8 	mov	x24, x0
	for(sptIndex i=1; i<nmodes; ++i)
    178c:	7100067f 	cmp	w19, #0x1
    1790:	d2800020 	mov	x0, #0x1                   	// #1
	mats_order[0] = mode;
    1794:	b9000303 	str	w3, [x24]
	for(sptIndex i=1; i<nmodes; ++i)
    1798:	54000129 	b.ls	17bc <main+0x34c>  // b.plast
    179c:	d503201f 	nop
		mats_order[i] = (mode+i) % nmodes;
    17a0:	0b000062 	add	w2, w3, w0
    17a4:	1ad30841 	udiv	w1, w2, w19
    17a8:	1b138821 	msub	w1, w1, w19, w2
    17ac:	b8207b01 	str	w1, [x24, x0, lsl #2]
	for(sptIndex i=1; i<nmodes; ++i)
    17b0:	91000400 	add	x0, x0, #0x1
    17b4:	6b00027f 	cmp	w19, w0
    17b8:	54ffff48 	b.hi	17a0 <main+0x330>  // b.pmore

	/* For warm-up caches, timing not included */
	if(dev_id == -2) {
    17bc:	b9407be0 	ldr	w0, [sp, #120]
    17c0:	3100081f 	cmn	w0, #0x2
    17c4:	54001c00 	b.eq	1b44 <main+0x6d4>  // b.none
		nthreads = 1;
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
	} else if(dev_id == -1) {
    17c8:	3100041f 	cmn	w0, #0x1
    17cc:	54001c80 	b.eq	1b5c <main+0x6ec>  // b.none
		nthreads = 1;
    17d0:	5280003c 	mov	w28, #0x1                   	// #1
    17d4:	910203f7 	add	x23, sp, #0x80
#endif
	}


	sptTimer timer;
	sptNewTimer(&timer, 0);
    17d8:	aa1703e0 	mov	x0, x23
    17dc:	52800001 	mov	w1, #0x0                   	// #0
    17e0:	94000c5c 	bl	4950 <sptNewTimer>
	sptStartTimer(timer);
    17e4:	f94043e0 	ldr	x0, [sp, #128]
    17e8:	528000b7 	mov	w23, #0x5                   	// #5
    17ec:	94000c79 	bl	49d0 <sptStartTimer>

	for(int it=0; it<niters; ++it) {
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    17f0:	0f000400 	movi	v0.2s, #0x0
    17f4:	f9400340 	ldr	x0, [x26]
    17f8:	94000716 	bl	3450 <sptConstantMatrix>
    17fc:	35001440 	cbnz	w0, 1a84 <main+0x614>
		if(dev_id == -2) {
    1800:	b9407be0 	ldr	w0, [sp, #120]
    1804:	3100081f 	cmn	w0, #0x2
    1808:	54001320 	b.eq	1a6c <main+0x5fc>  // b.none
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
		} else if(dev_id == -1) {
    180c:	3100041f 	cmn	w0, #0x1
    1810:	540013e0 	b.eq	1a8c <main+0x61c>  // b.none
	for(int it=0; it<niters; ++it) {
    1814:	710006f7 	subs	w23, w23, #0x1
    1818:	54fffec1 	b.ne	17f0 <main+0x380>  // b.any
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
#endif
		}
	}

	sptStopTimer(timer);
    181c:	f94043e0 	ldr	x0, [sp, #128]
    1820:	94000c9c 	bl	4a90 <sptStopTimer>

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    1824:	f94043e0 	ldr	x0, [sp, #128]
    1828:	90000022 	adrp	x2, 5000 <sptBytesString+0x80>
    182c:	91134042 	add	x2, x2, #0x4d0
    1830:	528000a1 	mov	w1, #0x5                   	// #5
    1834:	94000cf7 	bl	4c10 <sptPrintAverageElapsedTime>
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1838:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    183c:	b94077e5 	ldr	w5, [sp, #116]
	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    1840:	1e604001 	fmov	d1, d0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1844:	1e630262 	ucvtf	d2, w19
    1848:	fd431403 	ldr	d3, [x0, #1576]
    184c:	f94053e0 	ldr	x0, [sp, #160]
    1850:	1e6300a4 	ucvtf	d4, w5
    1854:	9e630000 	ucvtf	d0, x0
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1858:	9b000360 	madd	x0, x27, x0, x0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    185c:	1e640842 	fmul	d2, d2, d4
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1860:	d37ef400 	lsl	x0, x0, #2
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1864:	1e630800 	fmul	d0, d0, d3
    1868:	1e600842 	fmul	d2, d2, d0
    186c:	1e611840 	fdiv	d0, d2, d1
	for (sptIndex m=0; m<nmodes; ++m) {
    1870:	34001a33 	cbz	w19, 1bb4 <main+0x744>
    1874:	51000661 	sub	w1, w19, #0x1
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1878:	f9404fe4 	ldr	x4, [sp, #152]
    187c:	7100243f 	cmp	w1, #0x9
    1880:	540002e9 	b.ls	18dc <main+0x46c>  // b.plast
    1884:	4f000404 	movi	v4.4s, #0x0
    1888:	aa0403e1 	mov	x1, x4
    188c:	53027e62 	lsr	w2, w19, #2
    1890:	4f000485 	movi	v5.4s, #0x4
    1894:	51000443 	sub	w3, w2, #0x1
    1898:	91004082 	add	x2, x4, #0x10
    189c:	4e040ca6 	dup	v6.4s, w5
    18a0:	8b235042 	add	x2, x2, w3, uxtw #4
    18a4:	d503201f 	nop
    18a8:	3cc10422 	ldr	q2, [x1], #16
    18ac:	4ea69c42 	mul	v2.4s, v2.4s, v6.4s
    18b0:	eb02003f 	cmp	x1, x2
    18b4:	2ea5c043 	umull	v3.2d, v2.2s, v5.2s
    18b8:	6ea58043 	umlal2	v3.2d, v2.4s, v5.4s
    18bc:	4ee38484 	add	v4.2d, v4.2d, v3.2d
	for (sptIndex m=0; m<nmodes; ++m) {
    18c0:	54ffff41 	b.ne	18a8 <main+0x438>  // b.any
    18c4:	5ef1b884 	addp	d4, v4.2d
    18c8:	f240067f 	tst	x19, #0x3
    18cc:	121e7677 	and	w23, w19, #0xfffffffc
    18d0:	4e083c81 	mov	x1, v4.d[0]
    18d4:	8b010000 	add	x0, x0, x1
    18d8:	54000740 	b.eq	19c0 <main+0x550>  // b.none
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18dc:	b8775881 	ldr	w1, [x4, w23, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18e0:	110006e2 	add	w2, w23, #0x1
    18e4:	6b13005f 	cmp	w2, w19
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18e8:	1b017ca1 	mul	w1, w5, w1
    18ec:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    18f0:	54000682 	b.cs	19c0 <main+0x550>  // b.hs, b.nlast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18f4:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18f8:	11000ae2 	add	w2, w23, #0x2
    18fc:	6b13005f 	cmp	w2, w19
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1900:	1b017ca1 	mul	w1, w5, w1
    1904:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1908:	540005c2 	b.cs	19c0 <main+0x550>  // b.hs, b.nlast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    190c:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1910:	11000ee2 	add	w2, w23, #0x3
    1914:	6b02027f 	cmp	w19, w2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1918:	1b017ca1 	mul	w1, w5, w1
    191c:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1920:	54000509 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1924:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1928:	110012e2 	add	w2, w23, #0x4
    192c:	6b02027f 	cmp	w19, w2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1930:	1b017ca1 	mul	w1, w5, w1
    1934:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1938:	54000449 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    193c:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1940:	110016e2 	add	w2, w23, #0x5
    1944:	6b02027f 	cmp	w19, w2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1948:	1b017ca1 	mul	w1, w5, w1
    194c:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1950:	54000389 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1954:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1958:	11001ae2 	add	w2, w23, #0x6
    195c:	6b02027f 	cmp	w19, w2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1960:	1b017ca1 	mul	w1, w5, w1
    1964:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1968:	540002c9 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    196c:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1970:	11001ee2 	add	w2, w23, #0x7
    1974:	6b02027f 	cmp	w19, w2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1978:	1b017ca1 	mul	w1, w5, w1
    197c:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1980:	54000209 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1984:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1988:	110022e2 	add	w2, w23, #0x8
    198c:	6b02027f 	cmp	w19, w2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1990:	1b017ca1 	mul	w1, w5, w1
    1994:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1998:	54000149 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    199c:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    19a0:	110026f7 	add	w23, w23, #0x9
    19a4:	6b17027f 	cmp	w19, w23
		bytes += X.ndims[m] * R * sizeof(sptValue);
    19a8:	1b017ca1 	mul	w1, w5, w1
    19ac:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    19b0:	54000089 	b.ls	19c0 <main+0x550>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    19b4:	b8775881 	ldr	w1, [x4, w23, uxtw #2]
    19b8:	1b017ca1 	mul	w1, w5, w1
    19bc:	8b214800 	add	x0, x0, w1, uxtw #2
	}
	double gbw = (double)bytes / aver_time / 1e9;
    19c0:	90000021 	adrp	x1, 5000 <sptBytesString+0x80>
    19c4:	9e630002 	ucvtf	d2, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    19c8:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    19cc:	91168000 	add	x0, x0, #0x5a0
	double gbw = (double)bytes / aver_time / 1e9;
    19d0:	fd431423 	ldr	d3, [x1, #1576]
    19d4:	1e630842 	fmul	d2, d2, d3
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    19d8:	1e611841 	fdiv	d1, d2, d1
    19dc:	97fffe91 	bl	1420 <printf@plt>

	if(fo != NULL) {
    19e0:	b4000116 	cbz	x22, 1a00 <main+0x590>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    19e4:	f94037e0 	ldr	x0, [sp, #104]
    19e8:	aa1603e1 	mov	x1, x22
    19ec:	f8606a80 	ldr	x0, [x20, x0]
    19f0:	94000cc4 	bl	4d00 <sptDumpMatrix>
    19f4:	35000480 	cbnz	w0, 1a84 <main+0x614>
		fclose(fo);
    19f8:	aa1603e0 	mov	x0, x22
    19fc:	97fffe15 	bl	1250 <fclose@plt>
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1a00:	d2800016 	mov	x22, #0x0                   	// #0
	}

	sptFreeTimer(timer);
    1a04:	f94043e0 	ldr	x0, [sp, #128]
    1a08:	94000ca2 	bl	4c90 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1a0c:	d503201f 	nop
		sptFreeMatrix(U[m]);
    1a10:	f8767a80 	ldr	x0, [x20, x22, lsl #3]
	for(sptIndex m=0; m<nmodes; ++m) {
    1a14:	910006d6 	add	x22, x22, #0x1
		sptFreeMatrix(U[m]);
    1a18:	940006a2 	bl	34a0 <sptFreeMatrix>
	for(sptIndex m=0; m<nmodes; ++m) {
    1a1c:	6b16027f 	cmp	w19, w22
    1a20:	54ffff88 	b.hi	1a10 <main+0x5a0>  // b.pmore
	}
	sptFreeSparseTensor(&X);
    1a24:	aa1903e0 	mov	x0, x25
    1a28:	9400018a 	bl	2050 <sptFreeSparseTensor>
	free(mats_order);
    1a2c:	aa1803e0 	mov	x0, x24
    1a30:	97fffe64 	bl	13c0 <free@plt>
	sptFreeMatrix(U[nmodes]);
    1a34:	f87b7a80 	ldr	x0, [x20, x27, lsl #3]
    1a38:	9400069a 	bl	34a0 <sptFreeMatrix>
	free(U);
    1a3c:	aa1403e0 	mov	x0, x20
    1a40:	97fffe60 	bl	13c0 <free@plt>

	if (!random){
    1a44:	34000355 	cbz	w21, 1aac <main+0x63c>
		if(fPtr1 != NULL) fclose(fPtr1);
		if(fPtr2 != NULL) fclose(fPtr2);

	}
	return 0;
}
    1a48:	52800000 	mov	w0, #0x0                   	// #0
    1a4c:	a9407bfd 	ldp	x29, x30, [sp]
    1a50:	a94153f3 	ldp	x19, x20, [sp, #16]
    1a54:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1a58:	a94363f7 	ldp	x23, x24, [sp, #48]
    1a5c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    1a60:	a94573fb 	ldp	x27, x28, [sp, #80]
    1a64:	913203ff 	add	sp, sp, #0xc80
    1a68:	d65f03c0 	ret
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1a6c:	b94073e3 	ldr	w3, [sp, #112]
    1a70:	aa1803e2 	mov	x2, x24
    1a74:	aa1403e1 	mov	x1, x20
    1a78:	aa1903e0 	mov	x0, x25
    1a7c:	9400088d 	bl	3cb0 <sptMTTKRP>
    1a80:	34ffeca0 	cbz	w0, 1814 <main+0x3a4>
				sptAssert(fo != NULL);
    1a84:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a88:	97fffdd6 	bl	11e0 <exit@plt>
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1a8c:	b94073e3 	ldr	w3, [sp, #112]
    1a90:	2a1c03e4 	mov	w4, w28
    1a94:	aa1803e2 	mov	x2, x24
    1a98:	aa1403e1 	mov	x1, x20
    1a9c:	aa1903e0 	mov	x0, x25
    1aa0:	94000b30 	bl	4760 <sptOmpMTTKRP>
    1aa4:	34ffeb80 	cbz	w0, 1814 <main+0x3a4>
    1aa8:	17fffff7 	b	1a84 <main+0x614>
		FILE* fPtr1 = fopen(fvname, "r");
    1aac:	90000033 	adrp	x19, 5000 <sptBytesString+0x80>
    1ab0:	9113a273 	add	x19, x19, #0x4e8
    1ab4:	9112c3f6 	add	x22, sp, #0x4b0
    1ab8:	aa1303e1 	mov	x1, x19
    1abc:	aa1603e0 	mov	x0, x22
		FILE* fPtr2 = fopen(foname, "r");
    1ac0:	912263f5 	add	x21, sp, #0x898
		FILE* fPtr1 = fopen(fvname, "r");
    1ac4:	97fffde7 	bl	1260 <fopen@plt>
    1ac8:	aa0003f4 	mov	x20, x0
		FILE* fPtr2 = fopen(foname, "r");
    1acc:	aa1303e1 	mov	x1, x19
    1ad0:	aa1503e0 	mov	x0, x21
    1ad4:	97fffde3 	bl	1260 <fopen@plt>
		if (fPtr1 == NULL || fPtr2 == NULL) {
    1ad8:	f100029f 	cmp	x20, #0x0
		FILE* fPtr2 = fopen(foname, "r");
    1adc:	aa0003f3 	mov	x19, x0
		if (fPtr1 == NULL || fPtr2 == NULL) {
    1ae0:	fa401804 	ccmp	x0, #0x0, #0x4, ne  // ne = any
    1ae4:	54000200 	b.eq	1b24 <main+0x6b4>  // b.none
			int diff = compareFile(fPtr1, fPtr2);
    1ae8:	aa0003e1 	mov	x1, x0
    1aec:	aa1403e0 	mov	x0, x20
    1af0:	940000d0 	bl	1e30 <compareFile>
			if (diff == 0) {
    1af4:	35000820 	cbnz	w0, 1bf8 <main+0x788>
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
    1af8:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1afc:	aa1603e2 	mov	x2, x22
    1b00:	aa1503e1 	mov	x1, x21
    1b04:	91152000 	add	x0, x0, #0x548
    1b08:	97fffe46 	bl	1420 <printf@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1b0c:	aa1403e0 	mov	x0, x20
    1b10:	97fffdd0 	bl	1250 <fclose@plt>
		if(fPtr2 != NULL) fclose(fPtr2);
    1b14:	b4fff9b3 	cbz	x19, 1a48 <main+0x5d8>
    1b18:	aa1303e0 	mov	x0, x19
    1b1c:	97fffdcd 	bl	1250 <fclose@plt>
	return 0;
    1b20:	17ffffca 	b	1a48 <main+0x5d8>
			printf("\nUnable to open file.\n");
    1b24:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1b28:	9113c000 	add	x0, x0, #0x4f0
    1b2c:	97fffe15 	bl	1380 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    1b30:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1b34:	91142000 	add	x0, x0, #0x508
    1b38:	97fffe12 	bl	1380 <puts@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1b3c:	b4fffed4 	cbz	x20, 1b14 <main+0x6a4>
    1b40:	17fffff3 	b	1b0c <main+0x69c>
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1b44:	aa1803e2 	mov	x2, x24
    1b48:	aa1403e1 	mov	x1, x20
    1b4c:	aa1903e0 	mov	x0, x25
    1b50:	94000858 	bl	3cb0 <sptMTTKRP>
    1b54:	34ffe3e0 	cbz	w0, 17d0 <main+0x360>
    1b58:	17ffffcb 	b	1a84 <main+0x614>
		#pragma omp parallel
    1b5c:	52800022 	mov	w2, #0x1                   	// #1
    1b60:	910203f7 	add	x23, sp, #0x80
    1b64:	90000000 	adrp	x0, 1000 <g_last_error+0x1000>
    1b68:	52800003 	mov	w3, #0x0                   	// #0
    1b6c:	aa1703e1 	mov	x1, x23
    1b70:	91360000 	add	x0, x0, #0xd80
    1b74:	b90083e2 	str	w2, [sp, #128]
    1b78:	52800002 	mov	w2, #0x0                   	// #0
    1b7c:	97fffe31 	bl	1440 <GOMP_parallel@plt>
    1b80:	b94083fc 	ldr	w28, [sp, #128]
        printf("\nnthreads: %d\n", nthreads);
    1b84:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1b88:	91130000 	add	x0, x0, #0x4c0
    1b8c:	2a1c03e1 	mov	w1, w28
    1b90:	97fffe24 	bl	1420 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1b94:	b94073e3 	ldr	w3, [sp, #112]
    1b98:	2a1c03e4 	mov	w4, w28
    1b9c:	aa1803e2 	mov	x2, x24
    1ba0:	aa1403e1 	mov	x1, x20
    1ba4:	aa1903e0 	mov	x0, x25
    1ba8:	94000aee 	bl	4760 <sptOmpMTTKRP>
    1bac:	34ffe160 	cbz	w0, 17d8 <main+0x368>
    1bb0:	17ffffb5 	b	1a84 <main+0x614>
	double gbw = (double)bytes / aver_time / 1e9;
    1bb4:	9e630002 	ucvtf	d2, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1bb8:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1bbc:	91168000 	add	x0, x0, #0x5a0
	double gbw = (double)bytes / aver_time / 1e9;
    1bc0:	1e630842 	fmul	d2, d2, d3
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1bc4:	1e611841 	fdiv	d1, d2, d1
    1bc8:	97fffe16 	bl	1420 <printf@plt>
	if(fo != NULL) {
    1bcc:	b4000116 	cbz	x22, 1bec <main+0x77c>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1bd0:	f94037e0 	ldr	x0, [sp, #104]
    1bd4:	aa1603e1 	mov	x1, x22
    1bd8:	f8606a80 	ldr	x0, [x20, x0]
    1bdc:	94000c49 	bl	4d00 <sptDumpMatrix>
    1be0:	35fff520 	cbnz	w0, 1a84 <main+0x614>
		fclose(fo);
    1be4:	aa1603e0 	mov	x0, x22
    1be8:	97fffd9a 	bl	1250 <fclose@plt>
	sptFreeTimer(timer);
    1bec:	f94043e0 	ldr	x0, [sp, #128]
    1bf0:	94000c28 	bl	4c90 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1bf4:	17ffff8c 	b	1a24 <main+0x5b4>
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1bf8:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1bfc:	9115c000 	add	x0, x0, #0x570
    1c00:	97fffde0 	bl	1380 <puts@plt>
    1c04:	17ffffc2 	b	1b0c <main+0x69c>
				print_usage(argv);
    1c08:	f9400280 	ldr	x0, [x20]
    1c0c:	94000069 	bl	1db0 <print_usage.isra.0>
				exit(1);
    1c10:	52800020 	mov	w0, #0x1                   	// #1
    1c14:	97fffd73 	bl	11e0 <exit@plt>
    1c18:	d503201f 	nop
    1c1c:	d503201f 	nop

0000000000001c20 <init_have_lse_atomics>:
    1c20:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1c24:	d2800200 	mov	x0, #0x10                  	// #16
    1c28:	910003fd 	mov	x29, sp
    1c2c:	97fffdcd 	bl	1360 <__getauxval@plt>
    1c30:	53082000 	ubfx	w0, w0, #8, #1
    1c34:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1c38:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1c3c:	390aa420 	strb	w0, [x1, #681]
    1c40:	d65f03c0 	ret
    1c44:	d503201f 	nop
    1c48:	d503201f 	nop
    1c4c:	d503201f 	nop

0000000000001c50 <set_fast_math>:
    1c50:	52a02000 	mov	w0, #0x1000000             	// #16777216
    1c54:	d51b4400 	msr	fpcr, x0
    1c58:	d65f03c0 	ret

0000000000001c5c <_start>:
    1c5c:	d280001d 	mov	x29, #0x0                   	// #0
    1c60:	d280001e 	mov	x30, #0x0                   	// #0
    1c64:	aa0003e5 	mov	x5, x0
    1c68:	f94003e1 	ldr	x1, [sp]
    1c6c:	910023e2 	add	x2, sp, #0x8
    1c70:	910003e6 	mov	x6, sp
    1c74:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    1c78:	f947ec00 	ldr	x0, [x0, #4056]
    1c7c:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xf684>
    1c80:	f947e863 	ldr	x3, [x3, #4048]
    1c84:	b00000a4 	adrp	x4, 16000 <__FRAME_END__+0xf684>
    1c88:	f947cc84 	ldr	x4, [x4, #3992]
    1c8c:	97fffd85 	bl	12a0 <__libc_start_main@plt>
    1c90:	97fffdb8 	bl	1370 <abort@plt>

0000000000001c94 <call_weak_fn>:
    1c94:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    1c98:	f947e400 	ldr	x0, [x0, #4040]
    1c9c:	b4000040 	cbz	x0, 1ca4 <call_weak_fn+0x10>
    1ca0:	17fffdac 	b	1350 <__gmon_start__@plt>
    1ca4:	d65f03c0 	ret
    1ca8:	d503201f 	nop
    1cac:	d503201f 	nop

0000000000001cb0 <deregister_tm_clones>:
    1cb0:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1cb4:	910aa000 	add	x0, x0, #0x2a8
    1cb8:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1cbc:	910aa021 	add	x1, x1, #0x2a8
    1cc0:	eb00003f 	cmp	x1, x0
    1cc4:	540000c0 	b.eq	1cdc <deregister_tm_clones+0x2c>  // b.none
    1cc8:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xf684>
    1ccc:	f947d021 	ldr	x1, [x1, #4000]
    1cd0:	b4000061 	cbz	x1, 1cdc <deregister_tm_clones+0x2c>
    1cd4:	aa0103f0 	mov	x16, x1
    1cd8:	d61f0200 	br	x16
    1cdc:	d65f03c0 	ret

0000000000001ce0 <register_tm_clones>:
    1ce0:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1ce4:	910aa000 	add	x0, x0, #0x2a8
    1ce8:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1cec:	910aa021 	add	x1, x1, #0x2a8
    1cf0:	cb000021 	sub	x1, x1, x0
    1cf4:	d37ffc22 	lsr	x2, x1, #63
    1cf8:	8b810c41 	add	x1, x2, x1, asr #3
    1cfc:	9341fc21 	asr	x1, x1, #1
    1d00:	b40000c1 	cbz	x1, 1d18 <register_tm_clones+0x38>
    1d04:	b00000a2 	adrp	x2, 16000 <__FRAME_END__+0xf684>
    1d08:	f947f042 	ldr	x2, [x2, #4064]
    1d0c:	b4000062 	cbz	x2, 1d18 <register_tm_clones+0x38>
    1d10:	aa0203f0 	mov	x16, x2
    1d14:	d61f0200 	br	x16
    1d18:	d65f03c0 	ret
    1d1c:	d503201f 	nop

0000000000001d20 <__do_global_dtors_aux>:
    1d20:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1d24:	910003fd 	mov	x29, sp
    1d28:	f9000bf3 	str	x19, [sp, #16]
    1d2c:	d00000b3 	adrp	x19, 17000 <memcpy@GLIBC_2.17>
    1d30:	394aa260 	ldrb	w0, [x19, #680]
    1d34:	35000140 	cbnz	w0, 1d5c <__do_global_dtors_aux+0x3c>
    1d38:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    1d3c:	f947d400 	ldr	x0, [x0, #4008]
    1d40:	b4000080 	cbz	x0, 1d50 <__do_global_dtors_aux+0x30>
    1d44:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1d48:	f940ac00 	ldr	x0, [x0, #344]
    1d4c:	97fffd29 	bl	11f0 <__cxa_finalize@plt>
    1d50:	97ffffd8 	bl	1cb0 <deregister_tm_clones>
    1d54:	52800020 	mov	w0, #0x1                   	// #1
    1d58:	390aa260 	strb	w0, [x19, #680]
    1d5c:	f9400bf3 	ldr	x19, [sp, #16]
    1d60:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1d64:	d65f03c0 	ret
    1d68:	d503201f 	nop
    1d6c:	d503201f 	nop

0000000000001d70 <frame_dummy>:
    1d70:	17ffffdc 	b	1ce0 <register_tm_clones>
    1d74:	d503201f 	nop
    1d78:	d503201f 	nop
    1d7c:	d503201f 	nop

0000000000001d80 <main._omp_fn.0>:
		#pragma omp parallel
    1d80:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1d84:	910003fd 	mov	x29, sp
    1d88:	f9000bf3 	str	x19, [sp, #16]
    1d8c:	aa0003f3 	mov	x19, x0
            nthreads = omp_get_num_threads();
    1d90:	97fffd64 	bl	1320 <omp_get_num_threads@plt>
    1d94:	b9000260 	str	w0, [x19]
		#pragma omp parallel
    1d98:	f9400bf3 	ldr	x19, [sp, #16]
    1d9c:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1da0:	d65f03c0 	ret
    1da4:	d503201f 	nop
    1da8:	d503201f 	nop
    1dac:	d503201f 	nop

0000000000001db0 <print_usage.isra.0>:
	printf("Usage: %s [options] \n\n", argv[0]);
    1db0:	90000022 	adrp	x2, 5000 <sptBytesString+0x80>
static void print_usage(char ** argv) {
    1db4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
	printf("Usage: %s [options] \n\n", argv[0]);
    1db8:	aa0003e1 	mov	x1, x0
static void print_usage(char ** argv) {
    1dbc:	910003fd 	mov	x29, sp
	printf("Usage: %s [options] \n\n", argv[0]);
    1dc0:	9107a040 	add	x0, x2, #0x1e8
    1dc4:	97fffd97 	bl	1420 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1dc8:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1dcc:	91080000 	add	x0, x0, #0x200
    1dd0:	97fffd6c 	bl	1380 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    1dd4:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1dd8:	9108c000 	add	x0, x0, #0x230
    1ddc:	97fffd69 	bl	1380 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1de0:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1de4:	9109a000 	add	x0, x0, #0x268
    1de8:	97fffd66 	bl	1380 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    1dec:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1df0:	910b4000 	add	x0, x0, #0x2d0
    1df4:	97fffd63 	bl	1380 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    1df8:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1dfc:	910c8000 	add	x0, x0, #0x320
    1e00:	97fffd60 	bl	1380 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    1e04:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1e08:	910d8000 	add	x0, x0, #0x360
    1e0c:	97fffd5d 	bl	1380 <puts@plt>
	printf("         --help\n");
    1e10:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1e14:	910fc000 	add	x0, x0, #0x3f0
    1e18:	97fffd5a 	bl	1380 <puts@plt>
}
    1e1c:	a8c17bfd 	ldp	x29, x30, [sp], #16
	printf("\n");
    1e20:	52800140 	mov	w0, #0xa                   	// #10
    1e24:	17fffd8b 	b	1450 <putchar@plt>
    1e28:	d503201f 	nop
    1e2c:	d503201f 	nop

0000000000001e30 <compareFile>:



int compareFile(FILE * fPtr1, FILE * fPtr2)
{
    1e30:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1e34:	910003fd 	mov	x29, sp
    1e38:	a90153f3 	stp	x19, x20, [sp, #16]
    1e3c:	aa0103f4 	mov	x20, x1
    1e40:	f90013f5 	str	x21, [sp, #32]
    1e44:	aa0003f5 	mov	x21, x0
    1e48:	14000006 	b	1e60 <compareFile+0x30>
		ch1 = fgetc(fPtr1);
		ch2 = fgetc(fPtr2);
		if (ch1 != ch2) {
			return -1;
		}
	} while (ch1 != EOF && ch2 != EOF);
    1e4c:	3100067f 	cmn	w19, #0x1
    1e50:	1a9f07e2 	cset	w2, ne  // ne = any
    1e54:	7100005f 	cmp	w2, #0x0
    1e58:	3a411a64 	ccmn	w19, #0x1, #0x4, ne  // ne = any
    1e5c:	540001a0 	b.eq	1e90 <compareFile+0x60>  // b.none
		ch1 = fgetc(fPtr1);
    1e60:	aa1503e0 	mov	x0, x21
    1e64:	97fffd13 	bl	12b0 <fgetc@plt>
    1e68:	2a0003f3 	mov	w19, w0
		ch2 = fgetc(fPtr2);
    1e6c:	aa1403e0 	mov	x0, x20
    1e70:	97fffd10 	bl	12b0 <fgetc@plt>
		if (ch1 != ch2) {
    1e74:	6b00027f 	cmp	w19, w0
    1e78:	54fffea0 	b.eq	1e4c <compareFile+0x1c>  // b.none
			return -1;
    1e7c:	12800000 	mov	w0, #0xffffffff            	// #-1
	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
		return 0;
	else
		return -1;
}
    1e80:	a94153f3 	ldp	x19, x20, [sp, #16]
    1e84:	f94013f5 	ldr	x21, [sp, #32]
    1e88:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1e8c:	d65f03c0 	ret
	if (ch1 == EOF && ch2 == EOF)
    1e90:	4b0203e0 	neg	w0, w2
}
    1e94:	a94153f3 	ldp	x19, x20, [sp, #16]
    1e98:	f94013f5 	ldr	x21, [sp, #32]
    1e9c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1ea0:	d65f03c0 	ret
    1ea4:	d503201f 	nop
    1ea8:	d503201f 	nop
    1eac:	d503201f 	nop

0000000000001eb0 <sptNewSparseTensor>:
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1eb0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    1eb4:	910003fd 	mov	x29, sp
    1eb8:	a90153f3 	stp	x19, x20, [sp, #16]
    1ebc:	2a0103f3 	mov	w19, w1
    1ec0:	a9025bf5 	stp	x21, x22, [sp, #32]
    1ec4:	aa0003f6 	mov	x22, x0
    1ec8:	aa0203f5 	mov	x21, x2
    1ecc:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex i;
	int result;
	tsr->nmodes = nmodes;
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1ed0:	d37e7e78 	ubfiz	x24, x19, #2, #32
	}
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
//	spt_CheckOSError(!tsr->ndims, "SpTns New");
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
	tsr->nnz = 0;
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1ed4:	52800317 	mov	w23, #0x18                  	// #24
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1ed8:	aa1803e0 	mov	x0, x24
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1edc:	9bb77c37 	umull	x23, w1, w23
	tsr->nmodes = nmodes;
    1ee0:	b90002c1 	str	w1, [x22]
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1ee4:	97fffce7 	bl	1280 <malloc@plt>
    1ee8:	f90006c0 	str	x0, [x22, #8]
	for(i = 0; i < nmodes; ++i) {
    1eec:	340007b3 	cbz	w19, 1fe0 <sptNewSparseTensor+0x130>
    1ef0:	51000661 	sub	w1, w19, #0x1
    1ef4:	7101043f 	cmp	w1, #0x41
    1ef8:	54000a69 	b.ls	2044 <sptNewSparseTensor+0x194>  // b.plast
    1efc:	90000021 	adrp	x1, 5000 <sptBytesString+0x80>
    1f00:	4f000482 	movi	v2.4s, #0x4
    1f04:	aa0003e3 	mov	x3, x0
    1f08:	53027e64 	lsr	w4, w19, #2
    1f0c:	8b245004 	add	x4, x0, w4, uxtw #4
    1f10:	3dc19c20 	ldr	q0, [x1, #1648]
    1f14:	d503201f 	nop
    1f18:	4ea01c01 	mov	v1.16b, v0.16b
    1f1c:	4ea28400 	add	v0.4s, v0.4s, v2.4s
		tsr->sortorder[i] = i;
    1f20:	3c810461 	str	q1, [x3], #16
	for(i = 0; i < nmodes; ++i) {
    1f24:	eb04007f 	cmp	x3, x4
    1f28:	54ffff81 	b.ne	1f18 <sptNewSparseTensor+0x68>  // b.any
    1f2c:	f240067f 	tst	x19, #0x3
    1f30:	121e7663 	and	w3, w19, #0xfffffffc
    1f34:	540000a0 	b.eq	1f48 <sptNewSparseTensor+0x98>  // b.none
		tsr->sortorder[i] = i;
    1f38:	b8235803 	str	w3, [x0, w3, uxtw #2]
	for(i = 0; i < nmodes; ++i) {
    1f3c:	11000463 	add	w3, w3, #0x1
    1f40:	6b03027f 	cmp	w19, w3
    1f44:	54ffffa8 	b.hi	1f38 <sptNewSparseTensor+0x88>  // b.pmore
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1f48:	aa1803e0 	mov	x0, x24
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1f4c:	d2800014 	mov	x20, #0x0                   	// #0
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1f50:	97fffccc 	bl	1280 <malloc@plt>
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1f54:	aa1503e1 	mov	x1, x21
    1f58:	aa1803e2 	mov	x2, x24
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1f5c:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1f60:	97fffc9c 	bl	11d0 <memcpy@plt>
	tsr->nnz = 0;
    1f64:	52800315 	mov	w21, #0x18                  	// #24
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1f68:	aa1703e0 	mov	x0, x23
	tsr->nnz = 0;
    1f6c:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1f70:	97fffcc4 	bl	1280 <malloc@plt>
    1f74:	9bb57e75 	umull	x21, w19, w21
    1f78:	f90012c0 	str	x0, [x22, #32]
//	spt_CheckOSError(!tsr->inds, "SpTns New");
	for(i = 0; i < nmodes; ++i) {
    1f7c:	14000003 	b	1f88 <sptNewSparseTensor+0xd8>
    1f80:	54000440 	b.eq	2008 <sptNewSparseTensor+0x158>  // b.none
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1f84:	f94012c0 	ldr	x0, [x22, #32]
    1f88:	8b140000 	add	x0, x0, x20
    1f8c:	d2800002 	mov	x2, #0x0                   	// #0
    1f90:	d2800001 	mov	x1, #0x0                   	// #0
	for(i = 0; i < nmodes; ++i) {
    1f94:	91006294 	add	x20, x20, #0x18
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1f98:	94000126 	bl	2430 <sptNewIndexVector>
	for(i = 0; i < nmodes; ++i) {
    1f9c:	eb15029f 	cmp	x20, x21
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1fa0:	2a0003f3 	mov	w19, w0
		spt_CheckError(result, "SpTns New", NULL);
    1fa4:	34fffee0 	cbz	w0, 1f80 <sptNewSparseTensor+0xd0>
    1fa8:	90000022 	adrp	x2, 5000 <sptBytesString+0x80>
    1fac:	90000020 	adrp	x0, 5000 <sptBytesString+0x80>
    1fb0:	9118c042 	add	x2, x2, #0x630
    1fb4:	91196000 	add	x0, x0, #0x658
    1fb8:	d2800004 	mov	x4, #0x0                   	// #0
    1fbc:	52800643 	mov	w3, #0x32                  	// #50
    1fc0:	2a1303e1 	mov	w1, w19
    1fc4:	94000bb3 	bl	4e90 <spt_ComplainError>
	}
	result = sptNewValueVector(&tsr->values, 0, 0);
	spt_CheckError(result, "SpTns New", NULL);
	return 0;
}
    1fc8:	2a1303e0 	mov	w0, w19
    1fcc:	a94153f3 	ldp	x19, x20, [sp, #16]
    1fd0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1fd4:	a94363f7 	ldp	x23, x24, [sp, #48]
    1fd8:	a8c47bfd 	ldp	x29, x30, [sp], #64
    1fdc:	d65f03c0 	ret
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1fe0:	aa1803e0 	mov	x0, x24
    1fe4:	97fffca7 	bl	1280 <malloc@plt>
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1fe8:	aa1803e2 	mov	x2, x24
    1fec:	aa1503e1 	mov	x1, x21
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1ff0:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1ff4:	97fffc77 	bl	11d0 <memcpy@plt>
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1ff8:	aa1703e0 	mov	x0, x23
	tsr->nnz = 0;
    1ffc:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    2000:	97fffca0 	bl	1280 <malloc@plt>
    2004:	f90012c0 	str	x0, [x22, #32]
	result = sptNewValueVector(&tsr->values, 0, 0);
    2008:	9100a2c0 	add	x0, x22, #0x28
    200c:	d2800002 	mov	x2, #0x0                   	// #0
    2010:	d2800001 	mov	x1, #0x0                   	// #0
    2014:	9400002f 	bl	20d0 <sptNewValueVector>
    2018:	2a0003f3 	mov	w19, w0
	spt_CheckError(result, "SpTns New", NULL);
    201c:	34fffd60 	cbz	w0, 1fc8 <sptNewSparseTensor+0x118>
    2020:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    2024:	f0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    2028:	9118c042 	add	x2, x2, #0x630
    202c:	91196000 	add	x0, x0, #0x658
    2030:	d2800004 	mov	x4, #0x0                   	// #0
    2034:	528006a3 	mov	w3, #0x35                  	// #53
    2038:	2a1303e1 	mov	w1, w19
    203c:	94000b95 	bl	4e90 <spt_ComplainError>
    2040:	17ffffe2 	b	1fc8 <sptNewSparseTensor+0x118>
	for(i = 0; i < nmodes; ++i) {
    2044:	52800003 	mov	w3, #0x0                   	// #0
    2048:	17ffffbc 	b	1f38 <sptNewSparseTensor+0x88>
    204c:	d503201f 	nop

0000000000002050 <sptFreeSparseTensor>:

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    2050:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2054:	910003fd 	mov	x29, sp
    2058:	a90153f3 	stp	x19, x20, [sp, #16]
    205c:	aa0003f4 	mov	x20, x0
	sptIndex i;
	for(i = 0; i < tsr->nmodes; ++i) {
    2060:	b9400000 	ldr	w0, [x0]
    2064:	340001a0 	cbz	w0, 2098 <sptFreeSparseTensor+0x48>
    2068:	52800013 	mov	w19, #0x0                   	// #0
    206c:	f90013f5 	str	x21, [sp, #32]
		sptFreeIndexVector(&tsr->inds[i]);
    2070:	52800315 	mov	w21, #0x18                  	// #24
    2074:	d503201f 	nop
    2078:	f9401280 	ldr	x0, [x20, #32]
    207c:	9bb50260 	umaddl	x0, w19, w21, x0
	for(i = 0; i < tsr->nmodes; ++i) {
    2080:	11000673 	add	w19, w19, #0x1
		sptFreeIndexVector(&tsr->inds[i]);
    2084:	9400017b 	bl	2670 <sptFreeIndexVector>
	for(i = 0; i < tsr->nmodes; ++i) {
    2088:	b9400280 	ldr	w0, [x20]
    208c:	6b13001f 	cmp	w0, w19
    2090:	54ffff48 	b.hi	2078 <sptFreeSparseTensor+0x28>  // b.pmore
    2094:	f94013f5 	ldr	x21, [sp, #32]
	}
	free(tsr->sortorder);
    2098:	f9400680 	ldr	x0, [x20, #8]
    209c:	97fffcc9 	bl	13c0 <free@plt>
	free(tsr->ndims);
    20a0:	f9400a80 	ldr	x0, [x20, #16]
    20a4:	97fffcc7 	bl	13c0 <free@plt>
	free(tsr->inds);
    20a8:	f9401280 	ldr	x0, [x20, #32]
    20ac:	97fffcc5 	bl	13c0 <free@plt>
	sptFreeValueVector(&tsr->values);
    20b0:	9100a280 	add	x0, x20, #0x28
    20b4:	940000db 	bl	2420 <sptFreeValueVector>
	tsr->nmodes = 0;
    20b8:	b900029f 	str	wzr, [x20]
}
    20bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    20c0:	a8c37bfd 	ldp	x29, x30, [sp], #48
    20c4:	d65f03c0 	ret
    20c8:	d503201f 	nop
    20cc:	d503201f 	nop

00000000000020d0 <sptNewValueVector>:
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    20d0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    20d4:	910003fd 	mov	x29, sp
    20d8:	aa0103e4 	mov	x4, x1
    20dc:	f100083f 	cmp	x1, #0x2
	if(cap < len) {
		cap = len;
	}
	if(cap < 2) {
    20e0:	d2800043 	mov	x3, #0x2                   	// #2
    20e4:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    20e8:	d2800021 	mov	x1, #0x1                   	// #1
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    20ec:	a90153f3 	stp	x19, x20, [sp, #16]
    20f0:	aa0003f3 	mov	x19, x0
    20f4:	eb02007f 	cmp	x3, x2
    20f8:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
	vec->data = malloc(cap * sizeof *vec->data);
    20fc:	d37ef440 	lsl	x0, x2, #2
	vec->cap = cap;
    2100:	a9000a64 	stp	x4, x2, [x19]
	vec->data = malloc(cap * sizeof *vec->data);
    2104:	97fffc77 	bl	12e0 <calloc@plt>
    2108:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "ValVec New");
    210c:	b40000a0 	cbz	x0, 2120 <sptNewValueVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    2110:	52800000 	mov	w0, #0x0                   	// #0
}
    2114:	a94153f3 	ldp	x19, x20, [sp, #16]
    2118:	a8c27bfd 	ldp	x29, x30, [sp], #32
    211c:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "ValVec New");
    2120:	97fffcc4 	bl	1430 <__errno_location@plt>
    2124:	aa0003f3 	mov	x19, x0
	return 0;
    2128:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "ValVec New");
    212c:	b9400274 	ldr	w20, [x19]
    2130:	3140429f 	cmn	w20, #0x10, lsl #12
    2134:	54ffff00 	b.eq	2114 <sptNewValueVector+0x44>  // b.none
    2138:	2a1403e0 	mov	w0, w20
    213c:	97fffc7d 	bl	1330 <strerror@plt>
    2140:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    2144:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    2148:	aa0003e4 	mov	x4, x0
    214c:	11404281 	add	w1, w20, #0x10, lsl #12
    2150:	911aa060 	add	x0, x3, #0x6a8
    2154:	911a0042 	add	x2, x2, #0x680
    2158:	528005c3 	mov	w3, #0x2e                  	// #46
    215c:	94000b4d 	bl	4e90 <spt_ComplainError>
    2160:	b9400260 	ldr	w0, [x19]
    2164:	11404000 	add	w0, w0, #0x10, lsl #12
    2168:	17ffffeb 	b	2114 <sptNewValueVector+0x44>
    216c:	d503201f 	nop

0000000000002170 <sptConstantValueVector>:
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2170:	f9400001 	ldr	x1, [x0]
    2174:	b4000781 	cbz	x1, 2264 <sptConstantValueVector+0xf4>
    2178:	d1000422 	sub	x2, x1, #0x1
		vec->data[i] = val;
    217c:	f9400803 	ldr	x3, [x0, #16]
    2180:	f1002c5f 	cmp	x2, #0xb
    2184:	54000749 	b.ls	226c <sptConstantValueVector+0xfc>  // b.plast
    2188:	d342fc22 	lsr	x2, x1, #2
    218c:	4e040401 	dup	v1.4s, v0.s[0]
    2190:	aa0303e0 	mov	x0, x3
    2194:	8b021062 	add	x2, x3, x2, lsl #4
    2198:	3c810401 	str	q1, [x0], #16
	for(sptNnzIndex i=0; i<vec->len; ++i)
    219c:	eb02001f 	cmp	x0, x2
    21a0:	54ffffc1 	b.ne	2198 <sptConstantValueVector+0x28>  // b.any
    21a4:	f240043f 	tst	x1, #0x3
    21a8:	927ef420 	and	x0, x1, #0xfffffffffffffffc
    21ac:	540005c0 	b.eq	2264 <sptConstantValueVector+0xf4>  // b.none
    21b0:	91000402 	add	x2, x0, #0x1
		vec->data[i] = val;
    21b4:	bc207860 	str	s0, [x3, x0, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    21b8:	eb01005f 	cmp	x2, x1
    21bc:	54000542 	b.cs	2264 <sptConstantValueVector+0xf4>  // b.hs, b.nlast
    21c0:	91000804 	add	x4, x0, #0x2
		vec->data[i] = val;
    21c4:	bc227860 	str	s0, [x3, x2, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    21c8:	eb04003f 	cmp	x1, x4
    21cc:	540004c9 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    21d0:	91000c02 	add	x2, x0, #0x3
		vec->data[i] = val;
    21d4:	bc247860 	str	s0, [x3, x4, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    21d8:	eb02003f 	cmp	x1, x2
    21dc:	54000449 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    21e0:	91001004 	add	x4, x0, #0x4
		vec->data[i] = val;
    21e4:	bc227860 	str	s0, [x3, x2, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    21e8:	eb04003f 	cmp	x1, x4
    21ec:	540003c9 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    21f0:	91001402 	add	x2, x0, #0x5
		vec->data[i] = val;
    21f4:	bc247860 	str	s0, [x3, x4, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    21f8:	eb02003f 	cmp	x1, x2
    21fc:	54000349 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    2200:	91001804 	add	x4, x0, #0x6
		vec->data[i] = val;
    2204:	bc227860 	str	s0, [x3, x2, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2208:	eb04003f 	cmp	x1, x4
    220c:	540002c9 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    2210:	91001c02 	add	x2, x0, #0x7
		vec->data[i] = val;
    2214:	bc247860 	str	s0, [x3, x4, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2218:	eb02003f 	cmp	x1, x2
    221c:	54000249 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    2220:	91002004 	add	x4, x0, #0x8
		vec->data[i] = val;
    2224:	bc227860 	str	s0, [x3, x2, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2228:	eb04003f 	cmp	x1, x4
    222c:	540001c9 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    2230:	91002402 	add	x2, x0, #0x9
		vec->data[i] = val;
    2234:	bc247860 	str	s0, [x3, x4, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2238:	eb02003f 	cmp	x1, x2
    223c:	54000149 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    2240:	91002804 	add	x4, x0, #0xa
		vec->data[i] = val;
    2244:	bc227860 	str	s0, [x3, x2, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2248:	eb04003f 	cmp	x1, x4
    224c:	540000c9 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
    2250:	91002c00 	add	x0, x0, #0xb
		vec->data[i] = val;
    2254:	bc247860 	str	s0, [x3, x4, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2258:	eb00003f 	cmp	x1, x0
    225c:	54000049 	b.ls	2264 <sptConstantValueVector+0xf4>  // b.plast
		vec->data[i] = val;
    2260:	bc207860 	str	s0, [x3, x0, lsl #2]
	return 0;
}
    2264:	52800000 	mov	w0, #0x0                   	// #0
    2268:	d65f03c0 	ret
	for(sptNnzIndex i=0; i<vec->len; ++i)
    226c:	d2800000 	mov	x0, #0x0                   	// #0
    2270:	17ffffd0 	b	21b0 <sptConstantValueVector+0x40>
    2274:	d503201f 	nop
    2278:	d503201f 	nop
    227c:	d503201f 	nop

0000000000002280 <sptAppendValueVector>:
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    2280:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2284:	910003fd 	mov	x29, sp
    2288:	a90153f3 	stp	x19, x20, [sp, #16]
    228c:	aa0003f3 	mov	x19, x0
	if(vec->cap <= vec->len) {
    2290:	a9400001 	ldp	x1, x0, [x0]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2294:	f9400a74 	ldr	x20, [x19, #16]
	if(vec->cap <= vec->len) {
    2298:	eb01001f 	cmp	x0, x1
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    229c:	fd001fe8 	str	d8, [sp, #56]
    22a0:	1e204008 	fmov	s8, s0
	if(vec->cap <= vec->len) {
    22a4:	54000129 	b.ls	22c8 <sptAppendValueVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "ValVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
	++vec->len;
    22a8:	91000422 	add	x2, x1, #0x1
	return 0;
    22ac:	52800000 	mov	w0, #0x0                   	// #0
	vec->data[vec->len] = value;
    22b0:	bc217a88 	str	s8, [x20, x1, lsl #2]
	++vec->len;
    22b4:	f9000262 	str	x2, [x19]
}
    22b8:	a94153f3 	ldp	x19, x20, [sp, #16]
    22bc:	fd401fe8 	ldr	d8, [sp, #56]
    22c0:	a8c47bfd 	ldp	x29, x30, [sp], #64
    22c4:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    22c8:	a9025bf5 	stp	x21, x22, [sp, #32]
    22cc:	8b400415 	add	x21, x0, x0, lsr #1
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    22d0:	aa1403e0 	mov	x0, x20
    22d4:	d37ef6a1 	lsl	x1, x21, #2
    22d8:	97fffc06 	bl	12f0 <realloc@plt>
    22dc:	aa0003f4 	mov	x20, x0
		spt_CheckOSError(!newdata, "ValVec Append");
    22e0:	b40000a0 	cbz	x0, 22f4 <sptAppendValueVector+0x74>
		vec->data = newdata;
    22e4:	a900d275 	stp	x21, x20, [x19, #8]
	vec->data[vec->len] = value;
    22e8:	f9400261 	ldr	x1, [x19]
		vec->data = newdata;
    22ec:	a9425bf5 	ldp	x21, x22, [sp, #32]
    22f0:	17ffffee 	b	22a8 <sptAppendValueVector+0x28>
		spt_CheckOSError(!newdata, "ValVec Append");
    22f4:	f9001bf7 	str	x23, [sp, #48]
    22f8:	97fffc4e 	bl	1430 <__errno_location@plt>
    22fc:	b9400017 	ldr	w23, [x0]
    2300:	aa0003f6 	mov	x22, x0
    2304:	314042ff 	cmn	w23, #0x10, lsl #12
    2308:	54000061 	b.ne	2314 <sptAppendValueVector+0x94>  // b.any
    230c:	f9401bf7 	ldr	x23, [sp, #48]
    2310:	17fffff5 	b	22e4 <sptAppendValueVector+0x64>
    2314:	2a1703e0 	mov	w0, w23
    2318:	97fffc06 	bl	1330 <strerror@plt>
    231c:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    2320:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    2324:	aa0003e4 	mov	x4, x0
    2328:	114042e1 	add	w1, w23, #0x10, lsl #12
    232c:	911ae060 	add	x0, x3, #0x6b8
    2330:	911a0042 	add	x2, x2, #0x680
    2334:	52800a43 	mov	w3, #0x52                  	// #82
    2338:	94000ad6 	bl	4e90 <spt_ComplainError>
    233c:	b94002c0 	ldr	w0, [x22]
    2340:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2344:	11404000 	add	w0, w0, #0x10, lsl #12
    2348:	f9401bf7 	ldr	x23, [sp, #48]
    234c:	17ffffdb 	b	22b8 <sptAppendValueVector+0x38>

0000000000002350 <sptResizeValueVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2350:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2354:	910003fd 	mov	x29, sp
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2358:	f100083f 	cmp	x1, #0x2
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    235c:	a90153f3 	stp	x19, x20, [sp, #16]
    2360:	aa0003f3 	mov	x19, x0
    2364:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    2368:	f9400400 	ldr	x0, [x0, #8]
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    236c:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2370:	d2800055 	mov	x21, #0x2                   	// #2
    2374:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    2378:	eb15001f 	cmp	x0, x21
    237c:	540000e1 	b.ne	2398 <sptResizeValueVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    2380:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    2384:	f9000261 	str	x1, [x19]
}
    2388:	a94153f3 	ldp	x19, x20, [sp, #16]
    238c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2390:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2394:	d65f03c0 	ret
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2398:	d37ef6a1 	lsl	x1, x21, #2
    239c:	f9400a60 	ldr	x0, [x19, #16]
    23a0:	97fffbd4 	bl	12f0 <realloc@plt>
    23a4:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "ValVec Resize");
    23a8:	b4000100 	cbz	x0, 23c8 <sptResizeValueVector+0x78>
	return 0;
    23ac:	52800000 	mov	w0, #0x0                   	// #0
		vec->cap = newcap;
    23b0:	a9005674 	stp	x20, x21, [x19]
		vec->data = newdata;
    23b4:	f9000a76 	str	x22, [x19, #16]
}
    23b8:	a94153f3 	ldp	x19, x20, [sp, #16]
    23bc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    23c0:	a8c47bfd 	ldp	x29, x30, [sp], #64
    23c4:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "ValVec Resize");
    23c8:	a90363f7 	stp	x23, x24, [sp, #48]
    23cc:	97fffc19 	bl	1430 <__errno_location@plt>
    23d0:	b9400018 	ldr	w24, [x0]
    23d4:	aa0003f7 	mov	x23, x0
    23d8:	3140431f 	cmn	w24, #0x10, lsl #12
    23dc:	54000061 	b.ne	23e8 <sptResizeValueVector+0x98>  // b.any
    23e0:	a94363f7 	ldp	x23, x24, [sp, #48]
    23e4:	17fffff2 	b	23ac <sptResizeValueVector+0x5c>
    23e8:	2a1803e0 	mov	w0, w24
    23ec:	97fffbd1 	bl	1330 <strerror@plt>
    23f0:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    23f4:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    23f8:	aa0003e4 	mov	x4, x0
    23fc:	11404301 	add	w1, w24, #0x10, lsl #12
    2400:	911b2060 	add	x0, x3, #0x6c8
    2404:	911a0042 	add	x2, x2, #0x680
    2408:	52800d23 	mov	w3, #0x69                  	// #105
    240c:	94000aa1 	bl	4e90 <spt_ComplainError>
    2410:	b94002e0 	ldr	w0, [x23]
    2414:	a94363f7 	ldp	x23, x24, [sp, #48]
    2418:	11404000 	add	w0, w0, #0x10, lsl #12
    241c:	17ffffdb 	b	2388 <sptResizeValueVector+0x38>

0000000000002420 <sptFreeValueVector>:
 * Release the memory buffer a value vector is holding
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
    2420:	aa0003e1 	mov	x1, x0
	vec->len = 0;
	vec->cap = 0;
	free(vec->data);
    2424:	f9400800 	ldr	x0, [x0, #16]
	vec->cap = 0;
    2428:	a9007c3f 	stp	xzr, xzr, [x1]
	free(vec->data);
    242c:	17fffbe5 	b	13c0 <free@plt>

0000000000002430 <sptNewIndexVector>:
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    2430:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2434:	910003fd 	mov	x29, sp
    2438:	aa0103e4 	mov	x4, x1
    243c:	f100083f 	cmp	x1, #0x2
	if(cap < len) {
		cap = len;
	}
	if(cap < 2) {
    2440:	d2800043 	mov	x3, #0x2                   	// #2
    2444:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    2448:	d2800021 	mov	x1, #0x1                   	// #1
int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    244c:	a90153f3 	stp	x19, x20, [sp, #16]
    2450:	aa0003f3 	mov	x19, x0
    2454:	eb02007f 	cmp	x3, x2
    2458:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
	vec->data = malloc(cap * sizeof *vec->data);
    245c:	d37ef440 	lsl	x0, x2, #2
	vec->cap = cap;
    2460:	a9000a64 	stp	x4, x2, [x19]
	vec->data = malloc(cap * sizeof *vec->data);
    2464:	97fffb9f 	bl	12e0 <calloc@plt>
    2468:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "IdxVec New");
    246c:	b40000a0 	cbz	x0, 2480 <sptNewIndexVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    2470:	52800000 	mov	w0, #0x0                   	// #0
}
    2474:	a94153f3 	ldp	x19, x20, [sp, #16]
    2478:	a8c27bfd 	ldp	x29, x30, [sp], #32
    247c:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "IdxVec New");
    2480:	97fffbec 	bl	1430 <__errno_location@plt>
    2484:	aa0003f3 	mov	x19, x0
	return 0;
    2488:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "IdxVec New");
    248c:	b9400274 	ldr	w20, [x19]
    2490:	3140429f 	cmn	w20, #0x10, lsl #12
    2494:	54ffff00 	b.eq	2474 <sptNewIndexVector+0x44>  // b.none
    2498:	2a1403e0 	mov	w0, w20
    249c:	97fffba5 	bl	1330 <strerror@plt>
    24a0:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    24a4:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    24a8:	aa0003e4 	mov	x4, x0
    24ac:	11404281 	add	w1, w20, #0x10, lsl #12
    24b0:	911b6060 	add	x0, x3, #0x6d8
    24b4:	911a0042 	add	x2, x2, #0x680
    24b8:	52801283 	mov	w3, #0x94                  	// #148
    24bc:	94000a75 	bl	4e90 <spt_ComplainError>
    24c0:	b9400260 	ldr	w0, [x19]
    24c4:	11404000 	add	w0, w0, #0x10, lsl #12
    24c8:	17ffffeb 	b	2474 <sptNewIndexVector+0x44>
    24cc:	d503201f 	nop

00000000000024d0 <sptAppendIndexVector>:
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    24d0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    24d4:	910003fd 	mov	x29, sp
    24d8:	a90153f3 	stp	x19, x20, [sp, #16]
    24dc:	aa0003f3 	mov	x19, x0
    24e0:	2a0103f4 	mov	w20, w1
	if(vec->cap <= vec->len) {
    24e4:	a9400002 	ldp	x2, x0, [x0]
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    24e8:	a9025bf5 	stp	x21, x22, [sp, #32]
	if(vec->cap <= vec->len) {
    24ec:	eb02001f 	cmp	x0, x2
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    24f0:	f9400a75 	ldr	x21, [x19, #16]
	if(vec->cap <= vec->len) {
    24f4:	54000129 	b.ls	2518 <sptAppendIndexVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "IdxVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
	++vec->len;
    24f8:	91000441 	add	x1, x2, #0x1
	return 0;
    24fc:	52800000 	mov	w0, #0x0                   	// #0
	vec->data[vec->len] = value;
    2500:	b8227ab4 	str	w20, [x21, x2, lsl #2]
	++vec->len;
    2504:	f9000261 	str	x1, [x19]
}
    2508:	a94153f3 	ldp	x19, x20, [sp, #16]
    250c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2510:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2514:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    2518:	8b400416 	add	x22, x0, x0, lsr #1
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    251c:	aa1503e0 	mov	x0, x21
    2520:	d37ef6c1 	lsl	x1, x22, #2
    2524:	97fffb73 	bl	12f0 <realloc@plt>
    2528:	aa0003f5 	mov	x21, x0
		spt_CheckOSError(!newdata, "IdxVec Append");
    252c:	b4000080 	cbz	x0, 253c <sptAppendIndexVector+0x6c>
		vec->data = newdata;
    2530:	a900d676 	stp	x22, x21, [x19, #8]
	vec->data[vec->len] = value;
    2534:	f9400262 	ldr	x2, [x19]
    2538:	17fffff0 	b	24f8 <sptAppendIndexVector+0x28>
		spt_CheckOSError(!newdata, "IdxVec Append");
    253c:	a90363f7 	stp	x23, x24, [sp, #48]
    2540:	97fffbbc 	bl	1430 <__errno_location@plt>
    2544:	b9400018 	ldr	w24, [x0]
    2548:	aa0003f7 	mov	x23, x0
    254c:	3140431f 	cmn	w24, #0x10, lsl #12
    2550:	54000061 	b.ne	255c <sptAppendIndexVector+0x8c>  // b.any
    2554:	a94363f7 	ldp	x23, x24, [sp, #48]
    2558:	17fffff6 	b	2530 <sptAppendIndexVector+0x60>
    255c:	2a1803e0 	mov	w0, w24
    2560:	97fffb74 	bl	1330 <strerror@plt>
    2564:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    2568:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    256c:	aa0003e4 	mov	x4, x0
    2570:	11404301 	add	w1, w24, #0x10, lsl #12
    2574:	911ba060 	add	x0, x3, #0x6e8
    2578:	911a0042 	add	x2, x2, #0x680
    257c:	52801543 	mov	w3, #0xaa                  	// #170
    2580:	94000a44 	bl	4e90 <spt_ComplainError>
    2584:	b94002e0 	ldr	w0, [x23]
    2588:	a94363f7 	ldp	x23, x24, [sp, #48]
    258c:	11404000 	add	w0, w0, #0x10, lsl #12
    2590:	17ffffde 	b	2508 <sptAppendIndexVector+0x38>
    2594:	d503201f 	nop
    2598:	d503201f 	nop
    259c:	d503201f 	nop

00000000000025a0 <sptResizeIndexVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    25a0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    25a4:	910003fd 	mov	x29, sp
	sptNnzIndex newcap = size < 2 ? 2 : size;
    25a8:	f100083f 	cmp	x1, #0x2
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    25ac:	a90153f3 	stp	x19, x20, [sp, #16]
    25b0:	aa0003f3 	mov	x19, x0
    25b4:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    25b8:	f9400400 	ldr	x0, [x0, #8]
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    25bc:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    25c0:	d2800055 	mov	x21, #0x2                   	// #2
    25c4:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    25c8:	eb15001f 	cmp	x0, x21
    25cc:	540000e1 	b.ne	25e8 <sptResizeIndexVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    25d0:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    25d4:	f9000261 	str	x1, [x19]
}
    25d8:	a94153f3 	ldp	x19, x20, [sp, #16]
    25dc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    25e0:	a8c47bfd 	ldp	x29, x30, [sp], #64
    25e4:	d65f03c0 	ret
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    25e8:	d37ef6a1 	lsl	x1, x21, #2
    25ec:	f9400a60 	ldr	x0, [x19, #16]
    25f0:	97fffb40 	bl	12f0 <realloc@plt>
    25f4:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "IdxVec Resize");
    25f8:	b4000100 	cbz	x0, 2618 <sptResizeIndexVector+0x78>
	return 0;
    25fc:	52800000 	mov	w0, #0x0                   	// #0
		vec->cap = newcap;
    2600:	a9005674 	stp	x20, x21, [x19]
		vec->data = newdata;
    2604:	f9000a76 	str	x22, [x19, #16]
}
    2608:	a94153f3 	ldp	x19, x20, [sp, #16]
    260c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2610:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2614:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "IdxVec Resize");
    2618:	a90363f7 	stp	x23, x24, [sp, #48]
    261c:	97fffb85 	bl	1430 <__errno_location@plt>
    2620:	b9400018 	ldr	w24, [x0]
    2624:	aa0003f7 	mov	x23, x0
    2628:	3140431f 	cmn	w24, #0x10, lsl #12
    262c:	54000061 	b.ne	2638 <sptResizeIndexVector+0x98>  // b.any
    2630:	a94363f7 	ldp	x23, x24, [sp, #48]
    2634:	17fffff2 	b	25fc <sptResizeIndexVector+0x5c>
    2638:	2a1803e0 	mov	w0, w24
    263c:	97fffb3d 	bl	1330 <strerror@plt>
    2640:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    2644:	f0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    2648:	aa0003e4 	mov	x4, x0
    264c:	11404301 	add	w1, w24, #0x10, lsl #12
    2650:	911be060 	add	x0, x3, #0x6f8
    2654:	911a0042 	add	x2, x2, #0x680
    2658:	52801823 	mov	w3, #0xc1                  	// #193
    265c:	94000a0d 	bl	4e90 <spt_ComplainError>
    2660:	b94002e0 	ldr	w0, [x23]
    2664:	a94363f7 	ldp	x23, x24, [sp, #48]
    2668:	11404000 	add	w0, w0, #0x10, lsl #12
    266c:	17ffffdb 	b	25d8 <sptResizeIndexVector+0x38>

0000000000002670 <sptFreeIndexVector>:
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    2670:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2674:	910003fd 	mov	x29, sp
    2678:	f9000bf3 	str	x19, [sp, #16]
    267c:	aa0003f3 	mov	x19, x0
	free(vec->data);
    2680:	f9400800 	ldr	x0, [x0, #16]
    2684:	97fffb4f 	bl	13c0 <free@plt>
	vec->len = 0;
	vec->cap = 0;
    2688:	a9007e7f 	stp	xzr, xzr, [x19]
}
    268c:	f9400bf3 	ldr	x19, [sp, #16]
    2690:	a8c27bfd 	ldp	x29, x30, [sp], #32
    2694:	d65f03c0 	ret
    2698:	d503201f 	nop
    269c:	d503201f 	nop

00000000000026a0 <fill_binary_nnzidx._omp_fn.0>:
		sptIndex const BUF_LEN = 1024*1024;
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    26a0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    26a4:	910003fd 	mov	x29, sp
    26a8:	f90013f5 	str	x21, [sp, #32]
    26ac:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    26b0:	34000355 	cbz	w21, 2718 <fill_binary_nnzidx._omp_fn.0+0x78>
    26b4:	a90153f3 	stp	x19, x20, [sp, #16]
    26b8:	aa0003f4 	mov	x20, x0
    26bc:	97fffb19 	bl	1320 <omp_get_num_threads@plt>
    26c0:	2a0003f3 	mov	w19, w0
    26c4:	97fffadf 	bl	1240 <omp_get_thread_num@plt>
    26c8:	1ad30aa2 	udiv	w2, w21, w19
    26cc:	1b13d444 	msub	w4, w2, w19, w21
    26d0:	6b04001f 	cmp	w0, w4
    26d4:	54000283 	b.cc	2724 <fill_binary_nnzidx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    26d8:	1b001044 	madd	w4, w2, w0, w4
    26dc:	0b040042 	add	w2, w2, w4
    26e0:	6b02009f 	cmp	w4, w2
    26e4:	54000182 	b.cs	2714 <fill_binary_nnzidx._omp_fn.0+0x74>  // b.hs, b.nlast
#pragma omp parallel for schedule(static)
    26e8:	b9401281 	ldr	w1, [x20, #16]
    26ec:	a9400e85 	ldp	x5, x3, [x20]
    26f0:	8b244863 	add	x3, x3, w4, uxtw #2
    26f4:	0b020022 	add	w2, w1, w2
    26f8:	0b040020 	add	w0, w1, w4
    26fc:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2700:	b8404461 	ldr	w1, [x3], #4
    2704:	f82058a1 	str	x1, [x5, w0, uxtw #3]
    2708:	11000400 	add	w0, w0, #0x1
    270c:	6b00005f 	cmp	w2, w0
    2710:	54ffff81 	b.ne	2700 <fill_binary_nnzidx._omp_fn.0+0x60>  // b.any
    2714:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    2718:	f94013f5 	ldr	x21, [sp, #32]
    271c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2720:	d65f03c0 	ret
    2724:	11000442 	add	w2, w2, #0x1
    2728:	52800004 	mov	w4, #0x0                   	// #0
    272c:	17ffffeb 	b	26d8 <fill_binary_nnzidx._omp_fn.0+0x38>

0000000000002730 <fill_binary_idx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    2730:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2734:	910003fd 	mov	x29, sp
    2738:	f90013f5 	str	x21, [sp, #32]
    273c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2740:	34000355 	cbz	w21, 27a8 <fill_binary_idx._omp_fn.0+0x78>
    2744:	a90153f3 	stp	x19, x20, [sp, #16]
    2748:	aa0003f4 	mov	x20, x0
    274c:	97fffaf5 	bl	1320 <omp_get_num_threads@plt>
    2750:	2a0003f3 	mov	w19, w0
    2754:	97fffabb 	bl	1240 <omp_get_thread_num@plt>
    2758:	1ad30aa2 	udiv	w2, w21, w19
    275c:	1b13d444 	msub	w4, w2, w19, w21
    2760:	6b04001f 	cmp	w0, w4
    2764:	54000283 	b.cc	27b4 <fill_binary_idx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2768:	1b001044 	madd	w4, w2, w0, w4
    276c:	0b040042 	add	w2, w2, w4
    2770:	6b02009f 	cmp	w4, w2
    2774:	54000182 	b.cs	27a4 <fill_binary_idx._omp_fn.0+0x74>  // b.hs, b.nlast
#pragma omp parallel for schedule(static)
    2778:	b9401281 	ldr	w1, [x20, #16]
    277c:	a9400e85 	ldp	x5, x3, [x20]
    2780:	8b244863 	add	x3, x3, w4, uxtw #2
    2784:	0b020022 	add	w2, w1, w2
    2788:	0b040020 	add	w0, w1, w4
    278c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2790:	b8404461 	ldr	w1, [x3], #4
    2794:	b82058a1 	str	w1, [x5, w0, uxtw #2]
    2798:	11000400 	add	w0, w0, #0x1
    279c:	6b00005f 	cmp	w2, w0
    27a0:	54ffff81 	b.ne	2790 <fill_binary_idx._omp_fn.0+0x60>  // b.any
    27a4:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    27a8:	f94013f5 	ldr	x21, [sp, #32]
    27ac:	a8c37bfd 	ldp	x29, x30, [sp], #48
    27b0:	d65f03c0 	ret
    27b4:	11000442 	add	w2, w2, #0x1
    27b8:	52800004 	mov	w4, #0x0                   	// #0
    27bc:	17ffffeb 	b	2768 <fill_binary_idx._omp_fn.0+0x38>

00000000000027c0 <fill_binary_val._omp_fn.0>:
#endif

		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    27c0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    27c4:	910003fd 	mov	x29, sp
    27c8:	f90013f5 	str	x21, [sp, #32]
    27cc:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    27d0:	34000375 	cbz	w21, 283c <fill_binary_val._omp_fn.0+0x7c>
    27d4:	a90153f3 	stp	x19, x20, [sp, #16]
    27d8:	aa0003f4 	mov	x20, x0
    27dc:	97fffad1 	bl	1320 <omp_get_num_threads@plt>
    27e0:	2a0003f3 	mov	w19, w0
    27e4:	97fffa97 	bl	1240 <omp_get_thread_num@plt>
    27e8:	1ad30aa2 	udiv	w2, w21, w19
    27ec:	1b13d444 	msub	w4, w2, w19, w21
    27f0:	6b04001f 	cmp	w0, w4
    27f4:	540002a3 	b.cc	2848 <fill_binary_val._omp_fn.0+0x88>  // b.lo, b.ul, b.last
    27f8:	1b001044 	madd	w4, w2, w0, w4
    27fc:	0b040042 	add	w2, w2, w4
    2800:	6b02009f 	cmp	w4, w2
    2804:	540001a2 	b.cs	2838 <fill_binary_val._omp_fn.0+0x78>  // b.hs, b.nlast
#pragma omp parallel for schedule(static)
    2808:	b9401281 	ldr	w1, [x20, #16]
    280c:	a9400e85 	ldp	x5, x3, [x20]
    2810:	8b244c63 	add	x3, x3, w4, uxtw #3
    2814:	0b020022 	add	w2, w1, w2
    2818:	0b040020 	add	w0, w1, w4
    281c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2820:	fc408460 	ldr	d0, [x3], #8
    2824:	1e624000 	fcvt	s0, d0
    2828:	bc2058a0 	str	s0, [x5, w0, uxtw #2]
    282c:	11000400 	add	w0, w0, #0x1
    2830:	6b00005f 	cmp	w2, w0
    2834:	54ffff61 	b.ne	2820 <fill_binary_val._omp_fn.0+0x60>  // b.any
    2838:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    283c:	f94013f5 	ldr	x21, [sp, #32]
    2840:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2844:	d65f03c0 	ret
    2848:	11000442 	add	w2, w2, #0x1
    284c:	52800004 	mov	w4, #0x0                   	// #0
    2850:	17ffffea 	b	27f8 <fill_binary_val._omp_fn.0+0x38>
    2854:	d503201f 	nop
    2858:	d503201f 	nop
    285c:	d503201f 	nop

0000000000002860 <sptLoadSparseTensor>:
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname)
{
    2860:	a9b57bfd 	stp	x29, x30, [sp, #-176]!
    2864:	910003fd 	mov	x29, sp
    2868:	a90153f3 	stp	x19, x20, [sp, #16]
    286c:	aa0003f3 	mov	x19, x0
	FILE * fp = fopen(fname, "r");
    2870:	aa0203e0 	mov	x0, x2
{
    2874:	a90363f7 	stp	x23, x24, [sp, #48]
    2878:	2a0103f7 	mov	w23, w1
	FILE * fp = fopen(fname, "r");
    287c:	f0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    2880:	9113a021 	add	x1, x1, #0x4e8
{
    2884:	a9025bf5 	stp	x21, x22, [sp, #32]
    2888:	aa0203f5 	mov	x21, x2
	FILE * fp = fopen(fname, "r");
    288c:	97fffa75 	bl	1260 <fopen@plt>
	sptAssert(fp != NULL);
    2890:	b4004dc0 	cbz	x0, 3248 <sptLoadSparseTensor+0x9e8>
	char const * const suffix = strrchr(fname, '.');
    2894:	aa0003f4 	mov	x20, x0
    2898:	528005c1 	mov	w1, #0x2e                  	// #46
    289c:	aa1503e0 	mov	x0, x21
    28a0:	97fffaa8 	bl	1340 <strrchr@plt>
    28a4:	aa0003f6 	mov	x22, x0
	if(suffix == NULL) {
    28a8:	b4000200 	cbz	x0, 28e8 <sptLoadSparseTensor+0x88>
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    28ac:	f0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    28b0:	911c2021 	add	x1, x1, #0x708
    28b4:	97fffabb 	bl	13a0 <strcmp@plt>
    28b8:	34000a00 	cbz	w0, 29f8 <sptLoadSparseTensor+0x198>
    28bc:	f0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    28c0:	911c4021 	add	x1, x1, #0x710
    28c4:	aa1603e0 	mov	x0, x22
    28c8:	97fffab6 	bl	13a0 <strcmp@plt>
    28cc:	34003f00 	cbz	w0, 30ac <sptLoadSparseTensor+0x84c>
    28d0:	aa1603e0 	mov	x0, x22
    28d4:	f0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    28d8:	911c6021 	add	x1, x1, #0x718
    28dc:	97fffab1 	bl	13a0 <strcmp@plt>
	} while(file_extensions[++idx].extension != NULL);
    28e0:	d2800041 	mov	x1, #0x2                   	// #2
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    28e4:	340008c0 	cbz	w0, 29fc <sptLoadSparseTensor+0x19c>
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    28e8:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    28ec:	f0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    28f0:	a9046bf9 	stp	x25, x26, [sp, #64]
    28f4:	aa1503e2 	mov	x2, x21
    28f8:	911c8021 	add	x1, x1, #0x720
    28fc:	f947d800 	ldr	x0, [x0, #4016]
    2900:	f9400000 	ldr	x0, [x0]
    2904:	97fffad7 	bl	1460 <fprintf@plt>
	iores = fscanf(fp, "%u", &tsr->nmodes);
    2908:	f0000018 	adrp	x24, 5000 <sptBytesString+0x80>
    290c:	aa1303e2 	mov	x2, x19
    2910:	91112301 	add	x1, x24, #0x448
    2914:	aa1403e0 	mov	x0, x20
    2918:	97fffa5e 	bl	1290 <__isoc99_fscanf@plt>
	spt_CheckOSError(iores < 0, "SpTns Load");
    291c:	37f83820 	tbnz	w0, #31, 3020 <sptLoadSparseTensor+0x7c0>
	tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    2920:	b9400276 	ldr	w22, [x19]
    2924:	d2800021 	mov	x1, #0x1                   	// #1
    2928:	d37e7ed5 	ubfiz	x21, x22, #2, #32
    292c:	aa1503e0 	mov	x0, x21
    2930:	97fffa6c 	bl	12e0 <calloc@plt>
    2934:	f9000660 	str	x0, [x19, #8]
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    2938:	b4003f60 	cbz	x0, 3124 <sptLoadSparseTensor+0x8c4>
	tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    293c:	aa1503e0 	mov	x0, x21
    2940:	97fffa50 	bl	1280 <malloc@plt>
    2944:	f9000a60 	str	x0, [x19, #16]
    2948:	aa0003e2 	mov	x2, x0
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    294c:	b4003c20 	cbz	x0, 30d0 <sptLoadSparseTensor+0x870>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2950:	52800015 	mov	w21, #0x0                   	// #0
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2954:	91112319 	add	x25, x24, #0x448
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2958:	350000f6 	cbnz	w22, 2974 <sptLoadSparseTensor+0x114>
    295c:	1400020d 	b	3190 <sptLoadSparseTensor+0x930>
    2960:	b9400276 	ldr	w22, [x19]
    2964:	110006b5 	add	w21, w21, #0x1
    2968:	6b1602bf 	cmp	w21, w22
    296c:	54001722 	b.cs	2c50 <sptLoadSparseTensor+0x3f0>  // b.hs, b.nlast
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2970:	f9400a62 	ldr	x2, [x19, #16]
    2974:	aa1903e1 	mov	x1, x25
    2978:	aa1403e0 	mov	x0, x20
    297c:	8b354842 	add	x2, x2, w21, uxtw #2
    2980:	97fffa44 	bl	1290 <__isoc99_fscanf@plt>
		spt_CheckOSError(iores != 1, "SpTns Load");
    2984:	7100041f 	cmp	w0, #0x1
    2988:	54fffec0 	b.eq	2960 <sptLoadSparseTensor+0x100>  // b.none
    298c:	97fffaa9 	bl	1430 <__errno_location@plt>
    2990:	b940001a 	ldr	w26, [x0]
    2994:	aa0003f6 	mov	x22, x0
    2998:	3140435f 	cmn	w26, #0x10, lsl #12
    299c:	54fffe20 	b.eq	2960 <sptLoadSparseTensor+0x100>  // b.none
    29a0:	2a1a03e0 	mov	w0, w26
    29a4:	f0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    29a8:	97fffa62 	bl	1330 <strerror@plt>
    29ac:	f0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    29b0:	aa0003e4 	mov	x4, x0
    29b4:	11404341 	add	w1, w26, #0x10, lsl #12
    29b8:	911e82a0 	add	x0, x21, #0x7a0
    29bc:	911de262 	add	x2, x19, #0x778
    29c0:	52800a63 	mov	w3, #0x53                  	// #83
    29c4:	94000933 	bl	4e90 <spt_ComplainError>
    29c8:	b94002c0 	ldr	w0, [x22]
    29cc:	11404000 	add	w0, w0, #0x10, lsl #12

	int iores;
	switch(get_file_type(fname)) {
		case 0:
			iores = p_tt_read_file(tsr, start_index, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
    29d0:	350017a0 	cbnz	w0, 2cc4 <sptLoadSparseTensor+0x464>
    29d4:	a9446bf9 	ldp	x25, x26, [sp, #64]
			iores = p_tt_read_binary_file(tsr, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
			break;
	}

	fclose(fp);
    29d8:	aa1403e0 	mov	x0, x20
    29dc:	97fffa1d 	bl	1250 <fclose@plt>

	return 0;
    29e0:	52800000 	mov	w0, #0x0                   	// #0
}
    29e4:	a94153f3 	ldp	x19, x20, [sp, #16]
    29e8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    29ec:	a94363f7 	ldp	x23, x24, [sp, #48]
    29f0:	a8cb7bfd 	ldp	x29, x30, [sp], #176
    29f4:	d65f03c0 	ret
	size_t idx = 0;
    29f8:	d2800001 	mov	x1, #0x0                   	// #0
			return file_extensions[idx].type;
    29fc:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    2a00:	91350000 	add	x0, x0, #0xd40
    2a04:	8b011000 	add	x0, x0, x1, lsl #4
    2a08:	b9400800 	ldr	w0, [x0, #8]
	switch(get_file_type(fname)) {
    2a0c:	340034c0 	cbz	w0, 30a4 <sptLoadSparseTensor+0x844>
    2a10:	7100041f 	cmp	w0, #0x1
    2a14:	54fffe21 	b.ne	29d8 <sptLoadSparseTensor+0x178>  // b.any
	fread(&(header->magic), sizeof(header->magic), 1, fin);
    2a18:	aa1403e3 	mov	x3, x20
    2a1c:	d2800022 	mov	x2, #0x1                   	// #1
    2a20:	a9046bf9 	stp	x25, x26, [sp, #64]
    2a24:	d2800081 	mov	x1, #0x4                   	// #4
    2a28:	910203e0 	add	x0, sp, #0x80
    2a2c:	a90573fb 	stp	x27, x28, [sp, #80]
    2a30:	97fffa60 	bl	13b0 <fread@plt>
	fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
    2a34:	aa1403e3 	mov	x3, x20
    2a38:	d2800022 	mov	x2, #0x1                   	// #1
    2a3c:	d2800101 	mov	x1, #0x8                   	// #8
    2a40:	910223e0 	add	x0, sp, #0x88
    2a44:	97fffa5b 	bl	13b0 <fread@plt>
	fread(&(header->val_width), sizeof(header->val_width), 1, fin);
    2a48:	d2800101 	mov	x1, #0x8                   	// #8
    2a4c:	aa1403e3 	mov	x3, x20
    2a50:	d2800022 	mov	x2, #0x1                   	// #1
    2a54:	910243e0 	add	x0, sp, #0x90
    2a58:	97fffa56 	bl	13b0 <fread@plt>
	if(header->idx_width > PASTA_INDEX_TYPEWIDTH / 8) {
    2a5c:	f94047e1 	ldr	x1, [sp, #136]
    2a60:	f100103f 	cmp	x1, #0x4
    2a64:	54003fa8 	b.hi	3258 <sptLoadSparseTensor+0x9f8>  // b.pmore
	if(header->val_width > PASTA_VALUE_TYPEWIDTH / 8) {
    2a68:	f9404be0 	ldr	x0, [sp, #144]
    2a6c:	f100101f 	cmp	x0, #0x4
    2a70:	54002328 	b.hi	2ed4 <sptLoadSparseTensor+0x674>  // b.pmore
	if(header->idx_width == sizeof(sptIndex)) {
    2a74:	f100103f 	cmp	x1, #0x4
	sptIndex nmodes = 0;
    2a78:	b90077ff 	str	wzr, [sp, #116]
	sptNnzIndex nnz = 0;
    2a7c:	f9003fff 	str	xzr, [sp, #120]
	if(header->idx_width == sizeof(sptIndex)) {
    2a80:	54002200 	b.eq	2ec0 <sptLoadSparseTensor+0x660>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2a84:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2a88:	9000001b 	adrp	x27, 2000 <sptNewSparseTensor+0x150>
    2a8c:	97fff9fd 	bl	1280 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a90:	aa1403e3 	mov	x3, x20
    2a94:	d2800022 	mov	x2, #0x1                   	// #1
    2a98:	d2800081 	mov	x1, #0x4                   	// #4
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2a9c:	aa0003f5 	mov	x21, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2aa0:	97fffa44 	bl	13b0 <fread@plt>
#pragma omp parallel for schedule(static)
    2aa4:	9101d3e1 	add	x1, sp, #0x74
    2aa8:	d2c00024 	mov	x4, #0x100000000           	// #4294967296
    2aac:	911cc360 	add	x0, x27, #0x730
    2ab0:	52800003 	mov	w3, #0x0                   	// #0
    2ab4:	52800002 	mov	w2, #0x0                   	// #0
    2ab8:	a909d7e1 	stp	x1, x21, [sp, #152]
    2abc:	910263e1 	add	x1, sp, #0x98
    2ac0:	f90057e4 	str	x4, [sp, #168]
    2ac4:	97fffa5f 	bl	1440 <GOMP_parallel@plt>
		free(ubuf);
    2ac8:	aa1503e0 	mov	x0, x21
    2acc:	97fffa3d 	bl	13c0 <free@plt>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    2ad0:	b94077f7 	ldr	w23, [sp, #116]
    2ad4:	d37e7ee0 	ubfiz	x0, x23, #2, #32
    2ad8:	2a1703f6 	mov	w22, w23
    2adc:	97fff9e9 	bl	1280 <malloc@plt>
    2ae0:	f94047f5 	ldr	x21, [sp, #136]
    2ae4:	aa0003fa 	mov	x26, x0
	if(header->idx_width == sizeof(sptIndex)) {
    2ae8:	f10012bf 	cmp	x21, #0x4
    2aec:	54001de0 	b.eq	2ea8 <sptLoadSparseTensor+0x648>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2af0:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2af4:	97fff9e3 	bl	1280 <malloc@plt>
    2af8:	aa0003f8 	mov	x24, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2afc:	34000357 	cbz	w23, 2b64 <sptLoadSparseTensor+0x304>
    2b00:	9000001b 	adrp	x27, 2000 <sptNewSparseTensor+0x150>
    2b04:	52800016 	mov	w22, #0x0                   	// #0
    2b08:	911cc37b 	add	x27, x27, #0x730
    2b0c:	910263f9 	add	x25, sp, #0x98
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2b10:	52a0021c 	mov	w28, #0x100000              	// #1048576
    2b14:	d503201f 	nop
    2b18:	4b1602f5 	sub	w21, w23, w22
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2b1c:	aa1403e3 	mov	x3, x20
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2b20:	714402bf 	cmp	w21, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2b24:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2b28:	1a9c92b5 	csel	w21, w21, w28, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2b2c:	aa1803e0 	mov	x0, x24
    2b30:	2a1503e2 	mov	w2, w21
    2b34:	97fffa1f 	bl	13b0 <fread@plt>
    2b38:	52800003 	mov	w3, #0x0                   	// #0
    2b3c:	52800002 	mov	w2, #0x0                   	// #0
#pragma omp parallel for schedule(static)
    2b40:	a909e3fa 	stp	x26, x24, [sp, #152]
    2b44:	aa1903e1 	mov	x1, x25
    2b48:	aa1b03e0 	mov	x0, x27
    2b4c:	291557f6 	stp	w22, w21, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2b50:	114402d6 	add	w22, w22, #0x100, lsl #12
    2b54:	97fffa3b 	bl	1440 <GOMP_parallel@plt>
    2b58:	6b1602ff 	cmp	w23, w22
    2b5c:	54fffde8 	b.hi	2b18 <sptLoadSparseTensor+0x2b8>  // b.pmore
    2b60:	f94047f5 	ldr	x21, [sp, #136]
		free(ubuf);
    2b64:	aa1803e0 	mov	x0, x24
    2b68:	97fffa16 	bl	13c0 <free@plt>
	if(header->idx_width == sizeof(sptNnzIndex)) {
    2b6c:	f10022bf 	cmp	x21, #0x8
    2b70:	54001900 	b.eq	2e90 <sptLoadSparseTensor+0x630>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2b74:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2b78:	97fff9c2 	bl	1280 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2b7c:	aa1403e3 	mov	x3, x20
    2b80:	d2800022 	mov	x2, #0x1                   	// #1
    2b84:	d2800081 	mov	x1, #0x4                   	// #4
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2b88:	aa0003f5 	mov	x21, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2b8c:	97fffa09 	bl	13b0 <fread@plt>
#pragma omp parallel for schedule(static)
    2b90:	9101e3e1 	add	x1, sp, #0x78
    2b94:	d2c00024 	mov	x4, #0x100000000           	// #4294967296
    2b98:	90000000 	adrp	x0, 2000 <sptNewSparseTensor+0x150>
    2b9c:	52800003 	mov	w3, #0x0                   	// #0
    2ba0:	911a8000 	add	x0, x0, #0x6a0
    2ba4:	52800002 	mov	w2, #0x0                   	// #0
    2ba8:	a909d7e1 	stp	x1, x21, [sp, #152]
    2bac:	910263e1 	add	x1, sp, #0x98
    2bb0:	f90057e4 	str	x4, [sp, #168]
    2bb4:	97fffa23 	bl	1440 <GOMP_parallel@plt>
		free(ubuf);
    2bb8:	aa1503e0 	mov	x0, x21
    2bbc:	97fffa01 	bl	13c0 <free@plt>
	sptNewSparseTensor(tsr, nmodes, dims);
    2bc0:	b94077e1 	ldr	w1, [sp, #116]
    2bc4:	aa1a03e2 	mov	x2, x26
    2bc8:	aa1303e0 	mov	x0, x19
    2bcc:	97fffcb9 	bl	1eb0 <sptNewSparseTensor>
	tsr->nnz = nnz;
    2bd0:	f9403fe1 	ldr	x1, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2bd4:	b94077e0 	ldr	w0, [sp, #116]
	tsr->nnz = nnz;
    2bd8:	f9000e61 	str	x1, [x19, #24]
	for(sptIndex m=0; m < nmodes; ++m) {
    2bdc:	34000960 	cbz	w0, 2d08 <sptLoadSparseTensor+0x4a8>
    2be0:	52800015 	mov	w21, #0x0                   	// #0
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2be4:	52800316 	mov	w22, #0x18                  	// #24
    2be8:	14000006 	b	2c00 <sptLoadSparseTensor+0x3a0>
	for(sptIndex m=0; m < nmodes; ++m) {
    2bec:	b94077e0 	ldr	w0, [sp, #116]
    2bf0:	110006b5 	add	w21, w21, #0x1
	result = sptResizeValueVector(&tsr->values, nnz);
    2bf4:	f9403fe1 	ldr	x1, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2bf8:	6b0002bf 	cmp	w21, w0
    2bfc:	54000862 	b.cs	2d08 <sptLoadSparseTensor+0x4a8>  // b.hs, b.nlast
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2c00:	f9401260 	ldr	x0, [x19, #32]
    2c04:	9bb602a0 	umaddl	x0, w21, w22, x0
    2c08:	97fffe66 	bl	25a0 <sptResizeIndexVector>
		spt_CheckError(result, "SpTns Read", NULL);
    2c0c:	34ffff00 	cbz	w0, 2bec <sptLoadSparseTensor+0x38c>
    2c10:	f0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    2c14:	f0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    2c18:	2a0003e1 	mov	w1, w0
    2c1c:	911de262 	add	x2, x19, #0x778
    2c20:	91226060 	add	x0, x3, #0x898
    2c24:	d2800004 	mov	x4, #0x0                   	// #0
    2c28:	528020a3 	mov	w3, #0x105                 	// #261
    2c2c:	94000899 	bl	4e90 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2c30:	97fffa00 	bl	1430 <__errno_location@plt>
    2c34:	b9400016 	ldr	w22, [x0]
    2c38:	aa0003f5 	mov	x21, x0
    2c3c:	314042df 	cmn	w22, #0x10, lsl #12
    2c40:	54002161 	b.ne	306c <sptLoadSparseTensor+0x80c>  // b.any
    2c44:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2c48:	a94573fb 	ldp	x27, x28, [sp, #80]
    2c4c:	17ffff63 	b	29d8 <sptLoadSparseTensor+0x178>
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2c50:	52800300 	mov	w0, #0x18                  	// #24
	tsr->nnz = 0;
    2c54:	f9000e7f 	str	xzr, [x19, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2c58:	9ba07ec0 	umull	x0, w22, w0
    2c5c:	97fff989 	bl	1280 <malloc@plt>
    2c60:	f9001260 	str	x0, [x19, #32]
    2c64:	aa0003f9 	mov	x25, x0
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2c68:	b4002840 	cbz	x0, 3170 <sptLoadSparseTensor+0x910>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2c6c:	52800015 	mov	w21, #0x0                   	// #0
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2c70:	5280031a 	mov	w26, #0x18                  	// #24
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2c74:	350000f6 	cbnz	w22, 2c90 <sptLoadSparseTensor+0x430>
    2c78:	140000a2 	b	2f00 <sptLoadSparseTensor+0x6a0>
    2c7c:	b9400260 	ldr	w0, [x19]
    2c80:	110006b5 	add	w21, w21, #0x1
    2c84:	6b0002bf 	cmp	w21, w0
    2c88:	540013c2 	b.cs	2f00 <sptLoadSparseTensor+0x6a0>  // b.hs, b.nlast
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2c8c:	f9401279 	ldr	x25, [x19, #32]
    2c90:	9bba66a0 	umaddl	x0, w21, w26, x25
    2c94:	d2800002 	mov	x2, #0x0                   	// #0
    2c98:	d2800001 	mov	x1, #0x0                   	// #0
    2c9c:	97fffde5 	bl	2430 <sptNewIndexVector>
		spt_CheckError(retval, "SpTns Load", NULL);
    2ca0:	34fffee0 	cbz	w0, 2c7c <sptLoadSparseTensor+0x41c>
    2ca4:	f0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    2ca8:	f0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    2cac:	2a0003e1 	mov	w1, w0
    2cb0:	911de262 	add	x2, x19, #0x778
    2cb4:	911e82a0 	add	x0, x21, #0x7a0
    2cb8:	d2800004 	mov	x4, #0x0                   	// #0
    2cbc:	52800b43 	mov	w3, #0x5a                  	// #90
    2cc0:	94000874 	bl	4e90 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2cc4:	97fff9db 	bl	1430 <__errno_location@plt>
    2cc8:	b9400017 	ldr	w23, [x0]
    2ccc:	aa0003f6 	mov	x22, x0
    2cd0:	314042ff 	cmn	w23, #0x10, lsl #12
    2cd4:	54ffe800 	b.eq	29d4 <sptLoadSparseTensor+0x174>  // b.none
    2cd8:	2a1703e0 	mov	w0, w23
    2cdc:	97fff995 	bl	1330 <strerror@plt>
    2ce0:	aa0003e4 	mov	x4, x0
    2ce4:	911de262 	add	x2, x19, #0x778
    2ce8:	911e82a0 	add	x0, x21, #0x7a0
    2cec:	114042e1 	add	w1, w23, #0x10, lsl #12
    2cf0:	52802483 	mov	w3, #0x124                 	// #292
    2cf4:	94000867 	bl	4e90 <spt_ComplainError>
    2cf8:	b94002c0 	ldr	w0, [x22]
    2cfc:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2d00:	11404000 	add	w0, w0, #0x10, lsl #12
    2d04:	17ffff38 	b	29e4 <sptLoadSparseTensor+0x184>
	result = sptResizeValueVector(&tsr->values, nnz);
    2d08:	9100a260 	add	x0, x19, #0x28
    2d0c:	97fffd91 	bl	2350 <sptResizeValueVector>
	spt_CheckError(result, "SpTns Read", NULL);
    2d10:	35002780 	cbnz	w0, 3200 <sptLoadSparseTensor+0x9a0>
	for(sptIndex m=0; m < nmodes; ++m) {
    2d14:	b94077e0 	ldr	w0, [sp, #116]
    2d18:	b9006bff 	str	wzr, [sp, #104]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2d1c:	f9403ff8 	ldr	x24, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2d20:	340026c0 	cbz	w0, 31f8 <sptLoadSparseTensor+0x998>
    2d24:	90000019 	adrp	x25, 2000 <sptNewSparseTensor+0x150>
    2d28:	911cc339 	add	x25, x25, #0x730
    2d2c:	910263fa 	add	x26, sp, #0x98
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2d30:	b9406be2 	ldr	w2, [sp, #104]
    2d34:	52800303 	mov	w3, #0x18                  	// #24
    2d38:	f9401260 	ldr	x0, [x19, #32]
	if(header->idx_width == sizeof(sptIndex)) {
    2d3c:	f94047e1 	ldr	x1, [sp, #136]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2d40:	9ba30040 	umaddl	x0, w2, w3, x0
	if(header->idx_width == sizeof(sptIndex)) {
    2d44:	f100103f 	cmp	x1, #0x4
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2d48:	f9400817 	ldr	x23, [x0, #16]
	if(header->idx_width == sizeof(sptIndex)) {
    2d4c:	54000940 	b.eq	2e74 <sptLoadSparseTensor+0x614>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2d50:	d2a00800 	mov	x0, #0x400000              	// #4194304
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2d54:	2a1803f5 	mov	w21, w24
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2d58:	97fff94a 	bl	1280 <malloc@plt>
    2d5c:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2d60:	34000318 	cbz	w24, 2dc0 <sptLoadSparseTensor+0x560>
    2d64:	5280001c 	mov	w28, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2d68:	52a00218 	mov	w24, #0x100000              	// #1048576
    2d6c:	d503201f 	nop
    2d70:	4b1c02bb 	sub	w27, w21, w28
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2d74:	aa1403e3 	mov	x3, x20
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2d78:	7144037f 	cmp	w27, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2d7c:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2d80:	1a98937b 	csel	w27, w27, w24, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2d84:	aa1603e0 	mov	x0, x22
    2d88:	2a1b03e2 	mov	w2, w27
    2d8c:	97fff989 	bl	13b0 <fread@plt>
    2d90:	52800003 	mov	w3, #0x0                   	// #0
    2d94:	52800002 	mov	w2, #0x0                   	// #0
#pragma omp parallel for schedule(static)
    2d98:	a909dbf7 	stp	x23, x22, [sp, #152]
    2d9c:	aa1a03e1 	mov	x1, x26
    2da0:	aa1903e0 	mov	x0, x25
    2da4:	29156ffc 	stp	w28, w27, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2da8:	1144039c 	add	w28, w28, #0x100, lsl #12
    2dac:	97fff9a5 	bl	1440 <GOMP_parallel@plt>
    2db0:	6b1c02bf 	cmp	w21, w28
    2db4:	54fffde8 	b.hi	2d70 <sptLoadSparseTensor+0x510>  // b.pmore
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2db8:	f9403ff8 	ldr	x24, [sp, #120]
    2dbc:	2a1803f5 	mov	w21, w24
		free(ubuf);
    2dc0:	aa1603e0 	mov	x0, x22
    2dc4:	97fff97f 	bl	13c0 <free@plt>
	for(sptIndex m=0; m < nmodes; ++m) {
    2dc8:	b9406be1 	ldr	w1, [sp, #104]
    2dcc:	b94077e0 	ldr	w0, [sp, #116]
    2dd0:	11000421 	add	w1, w1, #0x1
    2dd4:	6b00003f 	cmp	w1, w0
    2dd8:	b9006be1 	str	w1, [sp, #104]
    2ddc:	54fffaa3 	b.cc	2d30 <sptLoadSparseTensor+0x4d0>  // b.lo, b.ul, b.last
	if(header->val_width == sizeof(sptValue)) {
    2de0:	f9404be1 	ldr	x1, [sp, #144]
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2de4:	f9401e7a 	ldr	x26, [x19, #56]
	if(header->val_width == sizeof(sptValue)) {
    2de8:	f100103f 	cmp	x1, #0x4
    2dec:	54001640 	b.eq	30b4 <sptLoadSparseTensor+0x854>  // b.none
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    2df0:	d2a01000 	mov	x0, #0x800000              	// #8388608
    2df4:	90000018 	adrp	x24, 2000 <sptNewSparseTensor+0x150>
    2df8:	97fff922 	bl	1280 <malloc@plt>
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2dfc:	52800016 	mov	w22, #0x0                   	// #0
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    2e00:	aa0003f7 	mov	x23, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2e04:	911f0318 	add	x24, x24, #0x7c0
    2e08:	910263f9 	add	x25, sp, #0x98
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2e0c:	52a0021b 	mov	w27, #0x100000              	// #1048576
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2e10:	34000295 	cbz	w21, 2e60 <sptLoadSparseTensor+0x600>
    2e14:	d503201f 	nop
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2e18:	4b1602b3 	sub	w19, w21, w22
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2e1c:	aa1403e3 	mov	x3, x20
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2e20:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2e24:	d2800101 	mov	x1, #0x8                   	// #8
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2e28:	1a9b9273 	csel	w19, w19, w27, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2e2c:	aa1703e0 	mov	x0, x23
    2e30:	2a1303e2 	mov	w2, w19
    2e34:	97fff95f 	bl	13b0 <fread@plt>
    2e38:	52800003 	mov	w3, #0x0                   	// #0
    2e3c:	52800002 	mov	w2, #0x0                   	// #0
#pragma omp parallel for schedule(static)
    2e40:	a909dffa 	stp	x26, x23, [sp, #152]
    2e44:	aa1903e1 	mov	x1, x25
    2e48:	aa1803e0 	mov	x0, x24
    2e4c:	29154ff6 	stp	w22, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2e50:	114402d6 	add	w22, w22, #0x100, lsl #12
    2e54:	97fff97b 	bl	1440 <GOMP_parallel@plt>
    2e58:	6b1502df 	cmp	w22, w21
    2e5c:	54fffde3 	b.cc	2e18 <sptLoadSparseTensor+0x5b8>  // b.lo, b.ul, b.last
		free(ubuf);
    2e60:	aa1703e0 	mov	x0, x23
    2e64:	97fff957 	bl	13c0 <free@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2e68:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2e6c:	a94573fb 	ldp	x27, x28, [sp, #80]
    2e70:	17fffeda 	b	29d8 <sptLoadSparseTensor+0x178>
		fread(buffer, sizeof(sptIndex), count, fin);
    2e74:	2a1803e2 	mov	w2, w24
    2e78:	aa1703e0 	mov	x0, x23
    2e7c:	aa1403e3 	mov	x3, x20
    2e80:	97fff94c 	bl	13b0 <fread@plt>
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2e84:	f9403ff8 	ldr	x24, [sp, #120]
    2e88:	2a1803f5 	mov	w21, w24
    2e8c:	17ffffcf 	b	2dc8 <sptLoadSparseTensor+0x568>
		fread(buffer, sizeof(sptNnzIndex), count, fin);
    2e90:	aa1503e1 	mov	x1, x21
    2e94:	aa1403e3 	mov	x3, x20
    2e98:	d2800022 	mov	x2, #0x1                   	// #1
    2e9c:	9101e3e0 	add	x0, sp, #0x78
    2ea0:	97fff944 	bl	13b0 <fread@plt>
    2ea4:	17ffff47 	b	2bc0 <sptLoadSparseTensor+0x360>
		fread(buffer, sizeof(sptIndex), count, fin);
    2ea8:	aa1503e1 	mov	x1, x21
    2eac:	aa1603e2 	mov	x2, x22
    2eb0:	aa1403e3 	mov	x3, x20
    2eb4:	97fff93f 	bl	13b0 <fread@plt>
    2eb8:	f94047f5 	ldr	x21, [sp, #136]
    2ebc:	17ffff2c 	b	2b6c <sptLoadSparseTensor+0x30c>
    2ec0:	aa1403e3 	mov	x3, x20
    2ec4:	d2800022 	mov	x2, #0x1                   	// #1
    2ec8:	9101d3e0 	add	x0, sp, #0x74
    2ecc:	97fff939 	bl	13b0 <fread@plt>
    2ed0:	17ffff00 	b	2ad0 <sptLoadSparseTensor+0x270>
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    2ed4:	900000a1 	adrp	x1, 16000 <__FRAME_END__+0xf684>
    2ed8:	d37df003 	lsl	x3, x0, #3
    2edc:	f947d820 	ldr	x0, [x1, #4016]
    2ee0:	f0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    2ee4:	aa0303e2 	mov	x2, x3
    2ee8:	91208021 	add	x1, x1, #0x820
    2eec:	f9400000 	ldr	x0, [x0]
    2ef0:	97fff95c 	bl	1460 <fprintf@plt>
    2ef4:	f94047e1 	ldr	x1, [sp, #136]
    2ef8:	17fffedf 	b	2a74 <sptLoadSparseTensor+0x214>
    2efc:	a94573fb 	ldp	x27, x28, [sp, #80]
	retval = sptNewValueVector(&tsr->values, 0, 0);
    2f00:	9100a27a 	add	x26, x19, #0x28
    2f04:	d2800002 	mov	x2, #0x0                   	// #0
    2f08:	aa1a03e0 	mov	x0, x26
    2f0c:	d2800001 	mov	x1, #0x0                   	// #0
    2f10:	97fffc70 	bl	20d0 <sptNewValueVector>
    2f14:	910263f9 	add	x25, sp, #0x98
	spt_CheckError(retval, "SpTns Load", NULL);
    2f18:	35001860 	cbnz	w0, 3224 <sptLoadSparseTensor+0x9c4>
			iores = fscanf(fp, "%u", &index);
    2f1c:	91112318 	add	x24, x24, #0x448
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2f20:	52800316 	mov	w22, #0x18                  	// #24
    2f24:	a90573fb 	stp	x27, x28, [sp, #80]
			iores = fscanf(fp, "%lf", &value);
    2f28:	f000001b 	adrp	x27, 5000 <sptBytesString+0x80>
    2f2c:	911f237b 	add	x27, x27, #0x7c8
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2f30:	52800015 	mov	w21, #0x0                   	// #0
    2f34:	b9400260 	ldr	w0, [x19]
    2f38:	350001a0 	cbnz	w0, 2f6c <sptLoadSparseTensor+0x70c>
    2f3c:	14000020 	b	2fbc <sptLoadSparseTensor+0x75c>
			if(index < start_index) {
    2f40:	b9409be1 	ldr	w1, [sp, #152]
    2f44:	6b0102ff 	cmp	w23, w1
    2f48:	54000568 	b.hi	2ff4 <sptLoadSparseTensor+0x794>  // b.pmore
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2f4c:	f9401260 	ldr	x0, [x19, #32]
    2f50:	4b170021 	sub	w1, w1, w23
    2f54:	9bb602a0 	umaddl	x0, w21, w22, x0
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2f58:	110006b5 	add	w21, w21, #0x1
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2f5c:	97fffd5d 	bl	24d0 <sptAppendIndexVector>
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2f60:	b9400260 	ldr	w0, [x19]
    2f64:	6b0002bf 	cmp	w21, w0
    2f68:	540002a2 	b.cs	2fbc <sptLoadSparseTensor+0x75c>  // b.hs, b.nlast
			iores = fscanf(fp, "%u", &index);
    2f6c:	aa1903e2 	mov	x2, x25
    2f70:	aa1803e1 	mov	x1, x24
    2f74:	aa1403e0 	mov	x0, x20
    2f78:	97fff8c6 	bl	1290 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2f7c:	7100041f 	cmp	w0, #0x1
    2f80:	54fffe00 	b.eq	2f40 <sptLoadSparseTensor+0x6e0>  // b.none
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2f84:	b9400260 	ldr	w0, [x19]
    2f88:	34ffe5e0 	cbz	w0, 2c44 <sptLoadSparseTensor+0x3e4>
    2f8c:	a9418a63 	ldp	x3, x2, [x19, #24]
    2f90:	51000400 	sub	w0, w0, #0x1
    2f94:	52800304 	mov	w4, #0x18                  	// #24
    2f98:	91006041 	add	x1, x2, #0x18
    2f9c:	9ba40400 	umaddl	x0, w0, w4, x1
    2fa0:	14000002 	b	2fa8 <sptLoadSparseTensor+0x748>
    2fa4:	91006021 	add	x1, x1, #0x18
    2fa8:	eb01001f 	cmp	x0, x1
		tsr->inds[mode].len = tsr->nnz;
    2fac:	f9000043 	str	x3, [x2]
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2fb0:	aa0103e2 	mov	x2, x1
    2fb4:	54ffff81 	b.ne	2fa4 <sptLoadSparseTensor+0x744>  // b.any
    2fb8:	17ffff23 	b	2c44 <sptLoadSparseTensor+0x3e4>
			iores = fscanf(fp, "%lf", &value);
    2fbc:	aa1903e2 	mov	x2, x25
    2fc0:	aa1b03e1 	mov	x1, x27
    2fc4:	aa1403e0 	mov	x0, x20
    2fc8:	97fff8b2 	bl	1290 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2fcc:	7100041f 	cmp	w0, #0x1
    2fd0:	54fffda1 	b.ne	2f84 <sptLoadSparseTensor+0x724>  // b.any
			sptAppendValueVector(&tsr->values, value);
    2fd4:	fd404fe0 	ldr	d0, [sp, #152]
    2fd8:	aa1a03e0 	mov	x0, x26
    2fdc:	1e624000 	fcvt	s0, d0
    2fe0:	97fffca8 	bl	2280 <sptAppendValueVector>
			++tsr->nnz;
    2fe4:	f9400e60 	ldr	x0, [x19, #24]
    2fe8:	91000400 	add	x0, x0, #0x1
    2fec:	f9000e60 	str	x0, [x19, #24]
	while(retval == 0) {
    2ff0:	17ffffd0 	b	2f30 <sptLoadSparseTensor+0x6d0>
				spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
    2ff4:	f0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    2ff8:	f0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    2ffc:	f0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    3000:	911ec084 	add	x4, x4, #0x7b0
    3004:	911de262 	add	x2, x19, #0x778
    3008:	911e82a0 	add	x0, x21, #0x7a0
    300c:	52800d03 	mov	w3, #0x68                  	// #104
    3010:	52800061 	mov	w1, #0x3                   	// #3
    3014:	9400079f 	bl	4e90 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    3018:	a94573fb 	ldp	x27, x28, [sp, #80]
    301c:	17ffff2a 	b	2cc4 <sptLoadSparseTensor+0x464>
	spt_CheckOSError(iores < 0, "SpTns Load");
    3020:	97fff904 	bl	1430 <__errno_location@plt>
    3024:	b9400019 	ldr	w25, [x0]
    3028:	aa0003f6 	mov	x22, x0
    302c:	3140433f 	cmn	w25, #0x10, lsl #12
    3030:	54ffc780 	b.eq	2920 <sptLoadSparseTensor+0xc0>  // b.none
    3034:	2a1903e0 	mov	w0, w25
    3038:	d0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    303c:	97fff8bd 	bl	1330 <strerror@plt>
    3040:	d0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    3044:	aa0003e4 	mov	x4, x0
    3048:	11404321 	add	w1, w25, #0x10, lsl #12
    304c:	911e82a0 	add	x0, x21, #0x7a0
    3050:	911de262 	add	x2, x19, #0x778
    3054:	52800943 	mov	w3, #0x4a                  	// #74
    3058:	9400078e 	bl	4e90 <spt_ComplainError>
    305c:	b94002c0 	ldr	w0, [x22]
    3060:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    3064:	34ffcb80 	cbz	w0, 29d4 <sptLoadSparseTensor+0x174>
    3068:	17ffff17 	b	2cc4 <sptLoadSparseTensor+0x464>
			spt_CheckOSError(iores != 0, "SpTns Load");
    306c:	2a1603e0 	mov	w0, w22
    3070:	97fff8b0 	bl	1330 <strerror@plt>
    3074:	d0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    3078:	aa0003e4 	mov	x4, x0
    307c:	911de262 	add	x2, x19, #0x778
    3080:	911e8060 	add	x0, x3, #0x7a0
    3084:	114042c1 	add	w1, w22, #0x10, lsl #12
    3088:	52802503 	mov	w3, #0x128                 	// #296
    308c:	94000781 	bl	4e90 <spt_ComplainError>
    3090:	b94002a0 	ldr	w0, [x21]
    3094:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3098:	a94573fb 	ldp	x27, x28, [sp, #80]
    309c:	11404000 	add	w0, w0, #0x10, lsl #12
    30a0:	17fffe51 	b	29e4 <sptLoadSparseTensor+0x184>
    30a4:	a9046bf9 	stp	x25, x26, [sp, #64]
    30a8:	17fffe18 	b	2908 <sptLoadSparseTensor+0xa8>
	} while(file_extensions[++idx].extension != NULL);
    30ac:	d2800021 	mov	x1, #0x1                   	// #1
    30b0:	17fffe53 	b	29fc <sptLoadSparseTensor+0x19c>
		fread(buffer, sizeof(sptValue), count, fin);
    30b4:	aa1a03e0 	mov	x0, x26
    30b8:	2a1803e2 	mov	w2, w24
    30bc:	aa1403e3 	mov	x3, x20
    30c0:	97fff8bc 	bl	13b0 <fread@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    30c4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    30c8:	a94573fb 	ldp	x27, x28, [sp, #80]
    30cc:	17fffe43 	b	29d8 <sptLoadSparseTensor+0x178>
    30d0:	f90037e0 	str	x0, [sp, #104]
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    30d4:	97fff8d7 	bl	1430 <__errno_location@plt>
    30d8:	b940001a 	ldr	w26, [x0]
    30dc:	aa0003f9 	mov	x25, x0
    30e0:	f94037e2 	ldr	x2, [sp, #104]
    30e4:	3140435f 	cmn	w26, #0x10, lsl #12
    30e8:	54ffc340 	b.eq	2950 <sptLoadSparseTensor+0xf0>  // b.none
    30ec:	2a1a03e0 	mov	w0, w26
    30f0:	d0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    30f4:	97fff88f 	bl	1330 <strerror@plt>
    30f8:	d0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    30fc:	aa0003e4 	mov	x4, x0
    3100:	11404341 	add	w1, w26, #0x10, lsl #12
    3104:	911e82a0 	add	x0, x21, #0x7a0
    3108:	911de262 	add	x2, x19, #0x778
    310c:	52800a03 	mov	w3, #0x50                  	// #80
    3110:	94000760 	bl	4e90 <spt_ComplainError>
    3114:	b9400320 	ldr	w0, [x25]
    3118:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    311c:	34ffc5c0 	cbz	w0, 29d4 <sptLoadSparseTensor+0x174>
    3120:	17fffee9 	b	2cc4 <sptLoadSparseTensor+0x464>
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    3124:	97fff8c3 	bl	1430 <__errno_location@plt>
    3128:	b940001a 	ldr	w26, [x0]
    312c:	aa0003f9 	mov	x25, x0
    3130:	3140435f 	cmn	w26, #0x10, lsl #12
    3134:	54ffc040 	b.eq	293c <sptLoadSparseTensor+0xdc>  // b.none
    3138:	2a1a03e0 	mov	w0, w26
    313c:	d0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    3140:	97fff87c 	bl	1330 <strerror@plt>
    3144:	d0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    3148:	aa0003e4 	mov	x4, x0
    314c:	11404341 	add	w1, w26, #0x10, lsl #12
    3150:	911e82a0 	add	x0, x21, #0x7a0
    3154:	911de262 	add	x2, x19, #0x778
    3158:	528009a3 	mov	w3, #0x4d                  	// #77
    315c:	9400074d 	bl	4e90 <spt_ComplainError>
    3160:	b9400320 	ldr	w0, [x25]
    3164:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    3168:	34ffc360 	cbz	w0, 29d4 <sptLoadSparseTensor+0x174>
    316c:	17fffed6 	b	2cc4 <sptLoadSparseTensor+0x464>
    3170:	a90573fb 	stp	x27, x28, [sp, #80]
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    3174:	97fff8af 	bl	1430 <__errno_location@plt>
    3178:	b940001b 	ldr	w27, [x0]
    317c:	aa0003fa 	mov	x26, x0
    3180:	3140437f 	cmn	w27, #0x10, lsl #12
    3184:	540001c1 	b.ne	31bc <sptLoadSparseTensor+0x95c>  // b.any
    3188:	a94573fb 	ldp	x27, x28, [sp, #80]
    318c:	17fffeb8 	b	2c6c <sptLoadSparseTensor+0x40c>
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    3190:	d2800000 	mov	x0, #0x0                   	// #0
	tsr->nnz = 0;
    3194:	f9000e7f 	str	xzr, [x19, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    3198:	97fff83a 	bl	1280 <malloc@plt>
    319c:	f9001260 	str	x0, [x19, #32]
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    31a0:	b5ffeb00 	cbnz	x0, 2f00 <sptLoadSparseTensor+0x6a0>
    31a4:	a90573fb 	stp	x27, x28, [sp, #80]
    31a8:	97fff8a2 	bl	1430 <__errno_location@plt>
    31ac:	b940001b 	ldr	w27, [x0]
    31b0:	aa0003fa 	mov	x26, x0
    31b4:	3140437f 	cmn	w27, #0x10, lsl #12
    31b8:	54ffea20 	b.eq	2efc <sptLoadSparseTensor+0x69c>  // b.none
    31bc:	2a1b03e0 	mov	w0, w27
    31c0:	d0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    31c4:	97fff85b 	bl	1330 <strerror@plt>
    31c8:	d0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    31cc:	aa0003e4 	mov	x4, x0
    31d0:	11404361 	add	w1, w27, #0x10, lsl #12
    31d4:	911e82a0 	add	x0, x21, #0x7a0
    31d8:	911de262 	add	x2, x19, #0x778
    31dc:	52800ae3 	mov	w3, #0x57                  	// #87
    31e0:	9400072c 	bl	4e90 <spt_ComplainError>
    31e4:	b9400340 	ldr	w0, [x26]
    31e8:	a94573fb 	ldp	x27, x28, [sp, #80]
    31ec:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    31f0:	34ffbf20 	cbz	w0, 29d4 <sptLoadSparseTensor+0x174>
    31f4:	17fffeb4 	b	2cc4 <sptLoadSparseTensor+0x464>
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    31f8:	2a1803f5 	mov	w21, w24
    31fc:	17fffef9 	b	2de0 <sptLoadSparseTensor+0x580>
	spt_CheckError(result, "SpTns Read", NULL);
    3200:	d0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    3204:	d0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    3208:	2a0003e1 	mov	w1, w0
    320c:	911de262 	add	x2, x19, #0x778
    3210:	91226060 	add	x0, x3, #0x898
    3214:	d2800004 	mov	x4, #0x0                   	// #0
    3218:	52802103 	mov	w3, #0x108                 	// #264
    321c:	9400071d 	bl	4e90 <spt_ComplainError>
    3220:	17fffe84 	b	2c30 <sptLoadSparseTensor+0x3d0>
	spt_CheckError(retval, "SpTns Load", NULL);
    3224:	d0000013 	adrp	x19, 5000 <sptBytesString+0x80>
    3228:	d0000015 	adrp	x21, 5000 <sptBytesString+0x80>
    322c:	2a0003e1 	mov	w1, w0
    3230:	911de262 	add	x2, x19, #0x778
    3234:	911e82a0 	add	x0, x21, #0x7a0
    3238:	d2800004 	mov	x4, #0x0                   	// #0
    323c:	52800ba3 	mov	w3, #0x5d                  	// #93
    3240:	94000714 	bl	4e90 <spt_ComplainError>
    3244:	17fffea0 	b	2cc4 <sptLoadSparseTensor+0x464>
	sptAssert(fp != NULL);
    3248:	12800000 	mov	w0, #0xffffffff            	// #-1
    324c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3250:	a90573fb 	stp	x27, x28, [sp, #80]
    3254:	97fff7e3 	bl	11e0 <exit@plt>
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    3258:	f0000080 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    325c:	d37df023 	lsl	x3, x1, #3
    3260:	f947d800 	ldr	x0, [x0, #4016]
    3264:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    3268:	aa0303e2 	mov	x2, x3
    326c:	911f4021 	add	x1, x1, #0x7d0
    3270:	f9400000 	ldr	x0, [x0]
    3274:	97fff87b 	bl	1460 <fprintf@plt>
		exit(-1);
    3278:	12800000 	mov	w0, #0xffffffff            	// #-1
    327c:	97fff7d9 	bl	11e0 <exit@plt>

0000000000003280 <sptNewMatrix>:
 */
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
	mtx->nrows = nrows;
	mtx->ncols = ncols;
	mtx->cap = nrows != 0 ? nrows : 1;
	mtx->stride = ((ncols-1)/8+1)*8;
    3280:	51000443 	sub	w3, w2, #0x1
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3284:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    3288:	910003fd 	mov	x29, sp
	mtx->stride = ((ncols-1)/8+1)*8;
    328c:	121d7063 	and	w3, w3, #0xfffffff8
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3290:	7100003f 	cmp	w1, #0x0
	mtx->stride = ((ncols-1)/8+1)*8;
    3294:	11002063 	add	w3, w3, #0x8
    3298:	1a9f1424 	csinc	w4, w1, wzr, ne  // ne = any
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    329c:	a90153f3 	stp	x19, x20, [sp, #16]
    32a0:	aa0003f3 	mov	x19, x0
#ifdef _ISOC11_SOURCE
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    32a4:	d2800400 	mov	x0, #0x20                  	// #32
    32a8:	1b047c74 	mul	w20, w3, w4
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    32ac:	a9025bf5 	stp	x21, x22, [sp, #32]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    32b0:	d37ef694 	lsl	x20, x20, #2
	mtx->ncols = ncols;
    32b4:	29000a61 	stp	w1, w2, [x19]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    32b8:	aa1403e1 	mov	x1, x20
	mtx->stride = ((ncols-1)/8+1)*8;
    32bc:	29010e64 	stp	w4, w3, [x19, #8]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    32c0:	97fff800 	bl	12c0 <aligned_alloc@plt>
    32c4:	f9000a60 	str	x0, [x19, #16]
    32c8:	aa0003f5 	mov	x21, x0
		}
	}
#else
	mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
	spt_CheckOSError(!mtx->values, "Mtx New");
    32cc:	b4000140 	cbz	x0, 32f4 <sptNewMatrix+0x74>
	memset(mtx->values, 0, mtx->cap * mtx->stride * sizeof (sptValue));
    32d0:	aa1403e2 	mov	x2, x20
    32d4:	aa1503e0 	mov	x0, x21
    32d8:	52800001 	mov	w1, #0x0                   	// #0
    32dc:	97fff7fd 	bl	12d0 <memset@plt>
	return 0;
    32e0:	52800000 	mov	w0, #0x0                   	// #0
}
    32e4:	a94153f3 	ldp	x19, x20, [sp, #16]
    32e8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    32ec:	a8c37bfd 	ldp	x29, x30, [sp], #48
    32f0:	d65f03c0 	ret
	spt_CheckOSError(!mtx->values, "Mtx New");
    32f4:	97fff84f 	bl	1430 <__errno_location@plt>
    32f8:	b9400016 	ldr	w22, [x0]
    32fc:	aa0003f3 	mov	x19, x0
    3300:	314042df 	cmn	w22, #0x10, lsl #12
    3304:	54fffe60 	b.eq	32d0 <sptNewMatrix+0x50>  // b.none
    3308:	2a1603e0 	mov	w0, w22
    330c:	97fff809 	bl	1330 <strerror@plt>
    3310:	d0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    3314:	d0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    3318:	aa0003e4 	mov	x4, x0
    331c:	114042c1 	add	w1, w22, #0x10, lsl #12
    3320:	91234060 	add	x0, x3, #0x8d0
    3324:	9122a042 	add	x2, x2, #0x8a8
    3328:	52800703 	mov	w3, #0x38                  	// #56
    332c:	940006d9 	bl	4e90 <spt_ComplainError>
    3330:	b9400260 	ldr	w0, [x19]
    3334:	11404000 	add	w0, w0, #0x10, lsl #12
    3338:	17ffffeb 	b	32e4 <sptNewMatrix+0x64>
    333c:	d503201f 	nop

0000000000003340 <sptRandomizeMatrix>:
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, bool random) {
    3340:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    3344:	910003fd 	mov	x29, sp
    3348:	a90153f3 	stp	x19, x20, [sp, #16]
    334c:	aa0003f3 	mov	x19, x0
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3350:	b9400000 	ldr	w0, [x0]
    3354:	34000440 	cbz	w0, 33dc <sptRandomizeMatrix+0x9c>
    3358:	52800014 	mov	w20, #0x0                   	// #0
    335c:	a9025bf5 	stp	x21, x22, [sp, #32]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
			if (random) {
				srand(time(NULL) + (rand()%100) + i + j);
    3360:	5290a3f6 	mov	w22, #0x851f                	// #34079
    3364:	72aa3d76 	movk	w22, #0x51eb, lsl #16
    3368:	52800c95 	mov	w21, #0x64                  	// #100
    336c:	a90363f7 	stp	x23, x24, [sp, #48]
    3370:	12001c37 	and	w23, w1, #0xff
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3374:	b9400660 	ldr	w0, [x19, #4]
    3378:	a9046bf9 	stp	x25, x26, [sp, #64]
    337c:	f9002bfb 	str	x27, [sp, #80]
    3380:	34000260 	cbz	w0, 33cc <sptRandomizeMatrix+0x8c>
    3384:	52800018 	mov	w24, #0x0                   	// #0
    3388:	35000337 	cbnz	w23, 33ec <sptRandomizeMatrix+0xac>
    338c:	11134a9a 	add	w26, w20, #0x4d2
			} else {
				srand(1234 + i + j);
    3390:	0b180340 	add	w0, w26, w24
    3394:	97fff81b 	bl	1400 <srand@plt>
			}
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3398:	b9400e79 	ldr	w25, [x19, #12]
    339c:	f9400a7b 	ldr	x27, [x19, #16]
    33a0:	94000748 	bl	50c0 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    33a4:	b9400660 	ldr	w0, [x19, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    33a8:	1b196299 	madd	w25, w20, w25, w24
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    33ac:	11000718 	add	w24, w24, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    33b0:	bc397b60 	str	s0, [x27, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    33b4:	6b00031f 	cmp	w24, w0
    33b8:	54fffec3 	b.cc	3390 <sptRandomizeMatrix+0x50>  // b.lo, b.ul, b.last
	for(sptIndex i=0; i<mtx->nrows; ++i)
    33bc:	b9400261 	ldr	w1, [x19]
    33c0:	11000694 	add	w20, w20, #0x1
    33c4:	6b01029f 	cmp	w20, w1
    33c8:	54fffdc3 	b.cc	3380 <sptRandomizeMatrix+0x40>  // b.lo, b.ul, b.last
    33cc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    33d0:	a94363f7 	ldp	x23, x24, [sp, #48]
    33d4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    33d8:	f9402bfb 	ldr	x27, [sp, #80]
		}
	return 0;
}
    33dc:	52800000 	mov	w0, #0x0                   	// #0
    33e0:	a94153f3 	ldp	x19, x20, [sp, #16]
    33e4:	a8c67bfd 	ldp	x29, x30, [sp], #96
    33e8:	d65f03c0 	ret
				srand(time(NULL) + (rand()%100) + i + j);
    33ec:	d2800000 	mov	x0, #0x0                   	// #0
    33f0:	0b18029a 	add	w26, w20, w24
    33f4:	97fff79f 	bl	1270 <time@plt>
    33f8:	aa0003f9 	mov	x25, x0
    33fc:	97fff7c1 	bl	1300 <rand@plt>
    3400:	9b367c01 	smull	x1, w0, w22
    3404:	9365fc21 	asr	x1, x1, #37
    3408:	4b807c21 	sub	w1, w1, w0, asr #31
    340c:	1b158021 	msub	w1, w1, w21, w0
    3410:	0b190020 	add	w0, w1, w25
    3414:	0b1a0000 	add	w0, w0, w26
    3418:	97fff7fa 	bl	1400 <srand@plt>
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    341c:	b9400e79 	ldr	w25, [x19, #12]
    3420:	f9400a7a 	ldr	x26, [x19, #16]
    3424:	94000727 	bl	50c0 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3428:	b9400660 	ldr	w0, [x19, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    342c:	1b196299 	madd	w25, w20, w25, w24
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3430:	11000718 	add	w24, w24, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3434:	bc397b40 	str	s0, [x26, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3438:	6b18001f 	cmp	w0, w24
    343c:	54fffd88 	b.hi	33ec <sptRandomizeMatrix+0xac>  // b.pmore
    3440:	17ffffdf 	b	33bc <sptRandomizeMatrix+0x7c>
    3444:	d503201f 	nop
    3448:	d503201f 	nop
    344c:	d503201f 	nop

0000000000003450 <sptConstantMatrix>:
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3450:	b9400004 	ldr	w4, [x0]
    3454:	34000204 	cbz	w4, 3494 <sptConstantMatrix+0x44>
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3458:	b9400405 	ldr	w5, [x0, #4]
    345c:	340001c5 	cbz	w5, 3494 <sptConstantMatrix+0x44>
			mtx->values[i * mtx->stride + j] = val;
    3460:	b9400c06 	ldr	w6, [x0, #12]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3464:	52800003 	mov	w3, #0x0                   	// #0
			mtx->values[i * mtx->stride + j] = val;
    3468:	f9400802 	ldr	x2, [x0, #16]
    346c:	d503201f 	nop
    3470:	1b037cc0 	mul	w0, w6, w3
    3474:	0b050001 	add	w1, w0, w5
    3478:	bc205840 	str	s0, [x2, w0, uxtw #2]
		for(sptIndex j=0; j<mtx->ncols; ++j)
    347c:	11000400 	add	w0, w0, #0x1
    3480:	6b00003f 	cmp	w1, w0
    3484:	54ffffa1 	b.ne	3478 <sptConstantMatrix+0x28>  // b.any
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3488:	11000463 	add	w3, w3, #0x1
    348c:	6b04007f 	cmp	w3, w4
    3490:	54ffff01 	b.ne	3470 <sptConstantMatrix+0x20>  // b.any
	return 0;
}
    3494:	52800000 	mov	w0, #0x0                   	// #0
    3498:	d65f03c0 	ret
    349c:	d503201f 	nop

00000000000034a0 <sptFreeMatrix>:
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    34a0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    34a4:	910003fd 	mov	x29, sp
    34a8:	f9000bf3 	str	x19, [sp, #16]
    34ac:	aa0003f3 	mov	x19, x0
	free(mtx->values);
    34b0:	f9400800 	ldr	x0, [x0, #16]
    34b4:	97fff7c3 	bl	13c0 <free@plt>
	mtx->nrows = 0;
	mtx->ncols = 0;
	mtx->cap = 0;
    34b8:	a9007e7f 	stp	xzr, xzr, [x19]
	mtx->stride = 0;
}
    34bc:	f9400bf3 	ldr	x19, [sp, #16]
    34c0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    34c4:	d65f03c0 	ret
    34c8:	d503201f 	nop
    34cc:	d503201f 	nop

00000000000034d0 <sptSparseTensorDensity>:
#include <bits/types/FILE.h>
#include <stdio.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
    34d0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    34d4:	910003fd 	mov	x29, sp
    34d8:	a90153f3 	stp	x19, x20, [sp, #16]
    34dc:	aa0003f4 	mov	x20, x0
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    34e0:	b9400013 	ldr	w19, [x0]
    34e4:	fd400c00 	ldr	d0, [x0, #24]
{
    34e8:	fd0013e8 	str	d8, [sp, #32]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    34ec:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
    34f0:	1e630261 	ucvtf	d1, w19
    34f4:	7e61d800 	ucvtf	d0, d0
    34f8:	1e611901 	fdiv	d1, d8, d1
    34fc:	97fff74d 	bl	1230 <pow@plt>
	double density = 1.0;
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3500:	34000db3 	cbz	w19, 36b4 <sptSparseTensorDensity+0x1e4>
    3504:	51000660 	sub	w0, w19, #0x1
    3508:	1e604005 	fmov	d5, d0
		density *= root / (double)tsr->ndims[m];
    350c:	f9400a82 	ldr	x2, [x20, #16]
    3510:	7100281f 	cmp	w0, #0xa
    3514:	54000da9 	b.ls	36c8 <sptSparseTensorDensity+0x1f8>  // b.plast
    3518:	4e080404 	dup	v4.2d, v0.d[0]
    351c:	6f03f603 	fmov	v3.2d, #1.000000000000000000e+00
    3520:	aa0203e0 	mov	x0, x2
    3524:	53027e61 	lsr	w1, w19, #2
    3528:	8b215041 	add	x1, x2, w1, uxtw #4
    352c:	d503201f 	nop
    3530:	3cc10401 	ldr	q1, [x0], #16
    3534:	2f20a422 	uxtl	v2.2d, v1.2s
    3538:	eb01001f 	cmp	x0, x1
    353c:	6f20a421 	uxtl2	v1.2d, v1.4s
    3540:	4e61d842 	scvtf	v2.2d, v2.2d
    3544:	4e61d821 	scvtf	v1.2d, v1.2d
    3548:	6e62fc82 	fdiv	v2.2d, v4.2d, v2.2d
    354c:	6e61fc81 	fdiv	v1.2d, v4.2d, v1.2d
    3550:	6e61dc41 	fmul	v1.2d, v2.2d, v1.2d
    3554:	6e61dc63 	fmul	v3.2d, v3.2d, v1.2d
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3558:	54fffec1 	b.ne	3530 <sptSparseTensorDensity+0x60>  // b.any
    355c:	6f00e401 	movi	v1.2d, #0x0
    3560:	f240067f 	tst	x19, #0x3
    3564:	121e7660 	and	w0, w19, #0xfffffffc
    3568:	6e014061 	ext	v1.16b, v3.16b, v1.16b, #8
    356c:	6e63dc21 	fmul	v1.2d, v1.2d, v3.2d
    3570:	5e080420 	mov	d0, v1.d[0]
    3574:	54000980 	b.eq	36a4 <sptSparseTensorDensity+0x1d4>  // b.none
		density *= root / (double)tsr->ndims[m];
    3578:	2a0003e1 	mov	w1, w0
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    357c:	11000403 	add	w3, w0, #0x1
		density *= root / (double)tsr->ndims[m];
    3580:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3584:	6b03027f 	cmp	w19, w3
		density *= root / (double)tsr->ndims[m];
    3588:	1e630021 	ucvtf	d1, w1
    358c:	1e6118a1 	fdiv	d1, d5, d1
    3590:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3594:	54000889 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3598:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    359c:	11000801 	add	w1, w0, #0x2
    35a0:	6b01027f 	cmp	w19, w1
		density *= root / (double)tsr->ndims[m];
    35a4:	1e630061 	ucvtf	d1, w3
    35a8:	1e6118a1 	fdiv	d1, d5, d1
    35ac:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35b0:	540007a9 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    35b4:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35b8:	11000c03 	add	w3, w0, #0x3
    35bc:	6b03027f 	cmp	w19, w3
		density *= root / (double)tsr->ndims[m];
    35c0:	1e630021 	ucvtf	d1, w1
    35c4:	1e6118a1 	fdiv	d1, d5, d1
    35c8:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35cc:	540006c9 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    35d0:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35d4:	11001001 	add	w1, w0, #0x4
    35d8:	6b01027f 	cmp	w19, w1
		density *= root / (double)tsr->ndims[m];
    35dc:	1e630061 	ucvtf	d1, w3
    35e0:	1e6118a1 	fdiv	d1, d5, d1
    35e4:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35e8:	540005e9 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    35ec:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35f0:	11001403 	add	w3, w0, #0x5
    35f4:	6b03027f 	cmp	w19, w3
		density *= root / (double)tsr->ndims[m];
    35f8:	1e630021 	ucvtf	d1, w1
    35fc:	1e6118a1 	fdiv	d1, d5, d1
    3600:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3604:	54000509 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3608:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    360c:	11001801 	add	w1, w0, #0x6
    3610:	6b01027f 	cmp	w19, w1
		density *= root / (double)tsr->ndims[m];
    3614:	1e630061 	ucvtf	d1, w3
    3618:	1e6118a1 	fdiv	d1, d5, d1
    361c:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3620:	54000429 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3624:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3628:	11001c03 	add	w3, w0, #0x7
    362c:	6b03027f 	cmp	w19, w3
		density *= root / (double)tsr->ndims[m];
    3630:	1e630021 	ucvtf	d1, w1
    3634:	1e6118a1 	fdiv	d1, d5, d1
    3638:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    363c:	54000349 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3640:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3644:	11002001 	add	w1, w0, #0x8
    3648:	6b01027f 	cmp	w19, w1
		density *= root / (double)tsr->ndims[m];
    364c:	1e630061 	ucvtf	d1, w3
    3650:	1e6118a1 	fdiv	d1, d5, d1
    3654:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3658:	54000269 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    365c:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3660:	11002403 	add	w3, w0, #0x9
    3664:	6b03027f 	cmp	w19, w3
		density *= root / (double)tsr->ndims[m];
    3668:	1e630021 	ucvtf	d1, w1
    366c:	1e6118a1 	fdiv	d1, d5, d1
    3670:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3674:	54000189 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3678:	b8635841 	ldr	w1, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    367c:	11002800 	add	w0, w0, #0xa
    3680:	6b00027f 	cmp	w19, w0
		density *= root / (double)tsr->ndims[m];
    3684:	1e630021 	ucvtf	d1, w1
    3688:	1e6118a1 	fdiv	d1, d5, d1
    368c:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3690:	540000a9 	b.ls	36a4 <sptSparseTensorDensity+0x1d4>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3694:	b8607840 	ldr	w0, [x2, x0, lsl #2]
    3698:	1e630001 	ucvtf	d1, w0
    369c:	1e6118a5 	fdiv	d5, d5, d1
    36a0:	1e650800 	fmul	d0, d0, d5
	}

	return density;
}
    36a4:	a94153f3 	ldp	x19, x20, [sp, #16]
    36a8:	fd4013e8 	ldr	d8, [sp, #32]
    36ac:	a8c37bfd 	ldp	x29, x30, [sp], #48
    36b0:	d65f03c0 	ret
	double density = 1.0;
    36b4:	1e604100 	fmov	d0, d8
}
    36b8:	a94153f3 	ldp	x19, x20, [sp, #16]
    36bc:	fd4013e8 	ldr	d8, [sp, #32]
    36c0:	a8c37bfd 	ldp	x29, x30, [sp], #48
    36c4:	d65f03c0 	ret
	double density = 1.0;
    36c8:	1e604100 	fmov	d0, d8
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    36cc:	52800000 	mov	w0, #0x0                   	// #0
    36d0:	17ffffaa 	b	3578 <sptSparseTensorDensity+0xa8>
    36d4:	d503201f 	nop
    36d8:	d503201f 	nop
    36dc:	d503201f 	nop

00000000000036e0 <sptSparseTensorStatus>:



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
    36e0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    36e4:	910003fd 	mov	x29, sp
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    36e8:	d2800822 	mov	x2, #0x41                  	// #65
{
    36ec:	a9025bf5 	stp	x21, x22, [sp, #32]
    36f0:	aa0103f5 	mov	x21, x1
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    36f4:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    36f8:	aa1503e3 	mov	x3, x21
    36fc:	d0000016 	adrp	x22, 5000 <sptBytesString+0x80>
    3700:	912502d6 	add	x22, x22, #0x940
{
    3704:	a90153f3 	stp	x19, x20, [sp, #16]
    3708:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    370c:	91236020 	add	x0, x1, #0x8d8
    3710:	d2800021 	mov	x1, #0x1                   	// #1
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3714:	52800034 	mov	w20, #0x1                   	// #1
{
    3718:	fd001be8 	str	d8, [sp, #48]
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    371c:	97fff72d 	bl	13d0 <fwrite@plt>
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3720:	f9400a62 	ldr	x2, [x19, #16]
    3724:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    3728:	aa1503e0 	mov	x0, x21
    372c:	91248021 	add	x1, x1, #0x920
    3730:	b9400042 	ldr	w2, [x2]
    3734:	97fff74b 	bl	1460 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3738:	b9400260 	ldr	w0, [x19]
    373c:	6b14001f 	cmp	w0, w20
    3740:	54000169 	b.ls	376c <sptSparseTensorStatus+0x8c>  // b.plast
    3744:	d503201f 	nop
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    3748:	f9400a62 	ldr	x2, [x19, #16]
    374c:	aa1603e1 	mov	x1, x22
    3750:	aa1503e0 	mov	x0, x21
    3754:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3758:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    375c:	97fff741 	bl	1460 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3760:	b9400260 	ldr	w0, [x19]
    3764:	6b14001f 	cmp	w0, w20
    3768:	54ffff08 	b.hi	3748 <sptSparseTensorStatus+0x68>  // b.pmore
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    376c:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
	}
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    3770:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    3774:	aa1503e0 	mov	x0, x21
    3778:	9124c021 	add	x1, x1, #0x930
    377c:	f9400e62 	ldr	x2, [x19, #24]
    3780:	97fff738 	bl	1460 <fprintf@plt>
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    3784:	b9400274 	ldr	w20, [x19]
    3788:	fd400e60 	ldr	d0, [x19, #24]
    378c:	1e630281 	ucvtf	d1, w20
    3790:	7e61d800 	ucvtf	d0, d0
    3794:	1e611901 	fdiv	d1, d8, d1
    3798:	97fff6a6 	bl	1230 <pow@plt>
    379c:	1e604005 	fmov	d5, d0
	double density = 1.0;
    37a0:	1e604100 	fmov	d0, d8
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    37a4:	34000d14 	cbz	w20, 3944 <sptSparseTensorStatus+0x264>
    37a8:	51000680 	sub	w0, w20, #0x1
		density *= root / (double)tsr->ndims[m];
    37ac:	f9400a62 	ldr	x2, [x19, #16]
    37b0:	7100281f 	cmp	w0, #0xa
    37b4:	54001489 	b.ls	3a44 <sptSparseTensorStatus+0x364>  // b.plast
    37b8:	4e0804a4 	dup	v4.2d, v5.d[0]
    37bc:	6f03f603 	fmov	v3.2d, #1.000000000000000000e+00
    37c0:	aa0203e0 	mov	x0, x2
    37c4:	53027e81 	lsr	w1, w20, #2
    37c8:	8b215041 	add	x1, x2, w1, uxtw #4
    37cc:	d503201f 	nop
    37d0:	3cc10401 	ldr	q1, [x0], #16
    37d4:	6f20a422 	uxtl2	v2.2d, v1.4s
    37d8:	eb01001f 	cmp	x0, x1
    37dc:	2f20a421 	uxtl	v1.2d, v1.2s
    37e0:	4e61d842 	scvtf	v2.2d, v2.2d
    37e4:	4e61d821 	scvtf	v1.2d, v1.2d
    37e8:	6e62fc82 	fdiv	v2.2d, v4.2d, v2.2d
    37ec:	6e61fc81 	fdiv	v1.2d, v4.2d, v1.2d
    37f0:	6e61dc41 	fmul	v1.2d, v2.2d, v1.2d
    37f4:	6e61dc63 	fmul	v3.2d, v3.2d, v1.2d
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    37f8:	54fffec1 	b.ne	37d0 <sptSparseTensorStatus+0xf0>  // b.any
    37fc:	6f00e401 	movi	v1.2d, #0x0
    3800:	f240069f 	tst	x20, #0x3
    3804:	121e7680 	and	w0, w20, #0xfffffffc
    3808:	6e014061 	ext	v1.16b, v3.16b, v1.16b, #8
    380c:	6e63dc21 	fmul	v1.2d, v1.2d, v3.2d
    3810:	5e080420 	mov	d0, v1.d[0]
    3814:	54000980 	b.eq	3944 <sptSparseTensorStatus+0x264>  // b.none
		density *= root / (double)tsr->ndims[m];
    3818:	2a0003e1 	mov	w1, w0
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    381c:	11000403 	add	w3, w0, #0x1
		density *= root / (double)tsr->ndims[m];
    3820:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3824:	6b03029f 	cmp	w20, w3
		density *= root / (double)tsr->ndims[m];
    3828:	1e630021 	ucvtf	d1, w1
    382c:	1e6118a1 	fdiv	d1, d5, d1
    3830:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3834:	54000889 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3838:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    383c:	11000801 	add	w1, w0, #0x2
    3840:	6b01029f 	cmp	w20, w1
		density *= root / (double)tsr->ndims[m];
    3844:	1e630061 	ucvtf	d1, w3
    3848:	1e6118a1 	fdiv	d1, d5, d1
    384c:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3850:	540007a9 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3854:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3858:	11000c03 	add	w3, w0, #0x3
    385c:	6b14007f 	cmp	w3, w20
		density *= root / (double)tsr->ndims[m];
    3860:	1e630021 	ucvtf	d1, w1
    3864:	1e6118a1 	fdiv	d1, d5, d1
    3868:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    386c:	540006c2 	b.cs	3944 <sptSparseTensorStatus+0x264>  // b.hs, b.nlast
		density *= root / (double)tsr->ndims[m];
    3870:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3874:	11001001 	add	w1, w0, #0x4
    3878:	6b01029f 	cmp	w20, w1
		density *= root / (double)tsr->ndims[m];
    387c:	1e630061 	ucvtf	d1, w3
    3880:	1e6118a1 	fdiv	d1, d5, d1
    3884:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3888:	540005e9 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    388c:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3890:	11001403 	add	w3, w0, #0x5
    3894:	6b03029f 	cmp	w20, w3
		density *= root / (double)tsr->ndims[m];
    3898:	1e630021 	ucvtf	d1, w1
    389c:	1e6118a1 	fdiv	d1, d5, d1
    38a0:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38a4:	54000509 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    38a8:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38ac:	11001801 	add	w1, w0, #0x6
    38b0:	6b01029f 	cmp	w20, w1
		density *= root / (double)tsr->ndims[m];
    38b4:	1e630061 	ucvtf	d1, w3
    38b8:	1e6118a1 	fdiv	d1, d5, d1
    38bc:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38c0:	54000429 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    38c4:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38c8:	11001c03 	add	w3, w0, #0x7
    38cc:	6b03029f 	cmp	w20, w3
		density *= root / (double)tsr->ndims[m];
    38d0:	1e630021 	ucvtf	d1, w1
    38d4:	1e6118a1 	fdiv	d1, d5, d1
    38d8:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38dc:	54000349 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    38e0:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38e4:	11002001 	add	w1, w0, #0x8
    38e8:	6b01029f 	cmp	w20, w1
		density *= root / (double)tsr->ndims[m];
    38ec:	1e630061 	ucvtf	d1, w3
    38f0:	1e6118a1 	fdiv	d1, d5, d1
    38f4:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    38f8:	54000269 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    38fc:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3900:	11002403 	add	w3, w0, #0x9
    3904:	6b03029f 	cmp	w20, w3
		density *= root / (double)tsr->ndims[m];
    3908:	1e630021 	ucvtf	d1, w1
    390c:	1e6118a1 	fdiv	d1, d5, d1
    3910:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3914:	54000189 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3918:	b8635841 	ldr	w1, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    391c:	11002800 	add	w0, w0, #0xa
    3920:	6b00029f 	cmp	w20, w0
		density *= root / (double)tsr->ndims[m];
    3924:	1e630021 	ucvtf	d1, w1
    3928:	1e6118a1 	fdiv	d1, d5, d1
    392c:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3930:	540000a9 	b.ls	3944 <sptSparseTensorStatus+0x264>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3934:	b8607840 	ldr	w0, [x2, x0, lsl #2]
    3938:	1e630001 	ucvtf	d1, w0
    393c:	1e6118a5 	fdiv	d5, d5, d1
    3940:	1e650800 	fmul	d0, d0, d5
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));
    3944:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    3948:	aa1503e0 	mov	x0, x21
    394c:	91252021 	add	x1, x1, #0x948
    3950:	97fff6c4 	bl	1460 <fprintf@plt>

	fprintf(fp, "Average slice length (c): ");
    3954:	d0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    3958:	aa1503e3 	mov	x3, x21
    395c:	91256000 	add	x0, x0, #0x958
    3960:	d2800342 	mov	x2, #0x1a                  	// #26
    3964:	d2800021 	mov	x1, #0x1                   	// #1
    3968:	97fff69a 	bl	13d0 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    396c:	b9400260 	ldr	w0, [x19]
    3970:	7100041f 	cmp	w0, #0x1
    3974:	54000640 	b.eq	3a3c <sptSparseTensorStatus+0x35c>  // b.none
    3978:	d0000016 	adrp	x22, 5000 <sptBytesString+0x80>
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    397c:	9125e2d6 	add	x22, x22, #0x978
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3980:	52800014 	mov	w20, #0x0                   	// #0
    3984:	d503201f 	nop
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3988:	f9400a62 	ldr	x2, [x19, #16]
    398c:	aa1603e1 	mov	x1, x22
    3990:	aa1503e0 	mov	x0, x21
    3994:	fd400e60 	ldr	d0, [x19, #24]
    3998:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    399c:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    39a0:	7e61d800 	ucvtf	d0, d0
    39a4:	1e630041 	ucvtf	d1, w2
    39a8:	1e611800 	fdiv	d0, d0, d1
    39ac:	97fff6ad 	bl	1460 <fprintf@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    39b0:	b9400262 	ldr	w2, [x19]
    39b4:	51000442 	sub	w2, w2, #0x1
    39b8:	6b14005f 	cmp	w2, w20
    39bc:	54fffe68 	b.hi	3988 <sptSparseTensorStatus+0x2a8>  // b.pmore
	}
	fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);
    39c0:	d37e7c42 	ubfiz	x2, x2, #2, #32
    39c4:	f9400a63 	ldr	x3, [x19, #16]
    39c8:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    39cc:	aa1503e0 	mov	x0, x21
    39d0:	91262021 	add	x1, x1, #0x988
    39d4:	fd400e60 	ldr	d0, [x19, #24]
    39d8:	b8626862 	ldr	w2, [x3, x2]
    39dc:	7e61d800 	ucvtf	d0, d0
    39e0:	1e630041 	ucvtf	d1, w2
    39e4:	1e611800 	fdiv	d0, d0, d1
    39e8:	97fff69e 	bl	1460 <fprintf@plt>

	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    39ec:	b9400261 	ldr	w1, [x19]
    39f0:	f9400e60 	ldr	x0, [x19, #24]
    39f4:	9b000020 	madd	x0, x1, x0, x0
    39f8:	d37ef400 	lsl	x0, x0, #2
    39fc:	94000561 	bl	4f80 <sptBytesString>
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3a00:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    3a04:	aa0003e2 	mov	x2, x0
	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    3a08:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3a0c:	91264021 	add	x1, x1, #0x990
    3a10:	aa1503e0 	mov	x0, x21
    3a14:	97fff693 	bl	1460 <fprintf@plt>
	fprintf(fp, "\n");
    3a18:	aa1503e1 	mov	x1, x21
    3a1c:	52800140 	mov	w0, #0xa                   	// #10
    3a20:	97fff5f8 	bl	1200 <fputc@plt>
	free(bytestr);
    3a24:	aa1303e0 	mov	x0, x19
}
    3a28:	a94153f3 	ldp	x19, x20, [sp, #16]
    3a2c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3a30:	fd401be8 	ldr	d8, [sp, #48]
    3a34:	a8c47bfd 	ldp	x29, x30, [sp], #64
	free(bytestr);
    3a38:	17fff662 	b	13c0 <free@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3a3c:	d2800002 	mov	x2, #0x0                   	// #0
    3a40:	17ffffe1 	b	39c4 <sptSparseTensorStatus+0x2e4>
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3a44:	52800000 	mov	w0, #0x0                   	// #0
    3a48:	17ffff74 	b	3818 <sptSparseTensorStatus+0x138>
    3a4c:	d503201f 	nop

0000000000003a50 <sptMTTKRP_3D>:

int sptMTTKRP_3D(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode)
{
    3a50:	a9b77bfd 	stp	x29, x30, [sp, #-144]!
    3a54:	910003fd 	mov	x29, sp
    3a58:	a9046bf9 	stp	x25, x26, [sp, #64]
    3a5c:	aa0103f9 	mov	x25, x1
    3a60:	aa0003fa 	mov	x26, x0
	sptIndex const * const ndims = X->ndims;
	sptValue const * const restrict vals = X->values.data;
	sptIndex const stride = mats[0]->stride;

	/* Check the mats. */
	sptAssert(nmodes ==3);
    3a64:	b9400001 	ldr	w1, [x0]
{
    3a68:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const stride = mats[0]->stride;
    3a6c:	f9400320 	ldr	x0, [x25]
	sptAssert(nmodes ==3);
    3a70:	71000c3f 	cmp	w1, #0x3
{
    3a74:	a9025bf5 	stp	x21, x22, [sp, #32]
    3a78:	a90363f7 	stp	x23, x24, [sp, #48]
    3a7c:	a90573fb 	stp	x27, x28, [sp, #80]
    3a80:	aa0203fb 	mov	x27, x2
	sptIndex const stride = mats[0]->stride;
    3a84:	b9400c14 	ldr	w20, [x0, #12]
	sptNnzIndex const nnz = X->nnz;
    3a88:	a9415742 	ldp	x2, x21, [x26, #16]
	sptValue const * const restrict vals = X->values.data;
    3a8c:	f9401f57 	ldr	x23, [x26, #56]
	sptAssert(nmodes ==3);
    3a90:	540010a1 	b.ne	3ca4 <sptMTTKRP_3D+0x254>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3a94:	f9400f25 	ldr	x5, [x25, #24]
    3a98:	b9400404 	ldr	w4, [x0, #4]
    3a9c:	b94004a1 	ldr	w1, [x5, #4]
    3aa0:	6b04003f 	cmp	w1, w4
    3aa4:	54000d41 	b.ne	3c4c <sptMTTKRP_3D+0x1fc>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
		}
		if(mats[i]->nrows != ndims[i]) {
    3aa8:	b9400000 	ldr	w0, [x0]
    3aac:	b9400044 	ldr	w4, [x2]
    3ab0:	6b00009f 	cmp	w4, w0
    3ab4:	54000e21 	b.ne	3c78 <sptMTTKRP_3D+0x228>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3ab8:	f9400720 	ldr	x0, [x25, #8]
    3abc:	b9400404 	ldr	w4, [x0, #4]
    3ac0:	6b04003f 	cmp	w1, w4
    3ac4:	54000c41 	b.ne	3c4c <sptMTTKRP_3D+0x1fc>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3ac8:	b9400000 	ldr	w0, [x0]
    3acc:	b9400444 	ldr	w4, [x2, #4]
    3ad0:	6b00009f 	cmp	w4, w0
    3ad4:	54000d21 	b.ne	3c78 <sptMTTKRP_3D+0x228>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3ad8:	f9400b20 	ldr	x0, [x25, #16]
    3adc:	b9400404 	ldr	w4, [x0, #4]
    3ae0:	6b01009f 	cmp	w4, w1
    3ae4:	54000b41 	b.ne	3c4c <sptMTTKRP_3D+0x1fc>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3ae8:	b9400001 	ldr	w1, [x0]
    3aec:	b9400840 	ldr	w0, [x2, #8]
    3af0:	6b00003f 	cmp	w1, w0
    3af4:	54000c21 	b.ne	3c78 <sptMTTKRP_3D+0x228>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}


	sptIndex const tmpI = mats[mode]->nrows;
    3af8:	f8635b20 	ldr	x0, [x25, w3, uxtw #3]
	sptIndex const R = mats[mode]->ncols;
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3afc:	5280031c 	mov	w28, #0x18                  	// #24
	sptMatrix * const restrict M = mats[nmodes];
	sptValue * const restrict mvals = M->values;
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3b00:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3b04:	f9401346 	ldr	x6, [x26, #32]
	sptValue * const restrict mvals = M->values;
    3b08:	f94008b3 	ldr	x19, [x5, #16]
	sptIndex const R = mats[mode]->ncols;
    3b0c:	29405802 	ldp	w2, w22, [x0]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3b10:	9bbc1863 	umaddl	x3, w3, w28, x6
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3b14:	aa1303e0 	mov	x0, x19
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3b18:	f9400878 	ldr	x24, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3b1c:	1b027e82 	mul	w2, w20, w2
    3b20:	fd0033e8 	str	d8, [sp, #96]
    3b24:	d37ef442 	lsl	x2, x2, #2
    3b28:	97fff5ea 	bl	12d0 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
	sptIndex times_mat_index_2 = mats_order[2];
    3b2c:	29408b63 	ldp	w3, w2, [x27, #4]
	sptIndex mode_i;
	sptIndex tmp_i_1, tmp_i_2;
	sptValue entry;

	sptTimer timer;
	sptNewTimer(&timer, 0);
    3b30:	52800001 	mov	w1, #0x0                   	// #0
    3b34:	910223e0 	add	x0, sp, #0x88
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3b38:	f9401345 	ldr	x5, [x26, #32]
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    3b3c:	f8635b24 	ldr	x4, [x25, w3, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3b40:	9bbc1463 	umaddl	x3, w3, w28, x5
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    3b44:	f8625b3a 	ldr	x26, [x25, w2, uxtw #3]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3b48:	9bbc1442 	umaddl	x2, w2, w28, x5
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    3b4c:	f9003fe4 	str	x4, [sp, #120]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3b50:	f940085b 	ldr	x27, [x2, #16]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3b54:	f9400879 	ldr	x25, [x3, #16]
	sptNewTimer(&timer, 0);
    3b58:	9400037e 	bl	4950 <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    3b5c:	f94047e0 	ldr	x0, [sp, #136]
    3b60:	9400039c 	bl	49d0 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3b64:	b4000455 	cbz	x21, 3bec <sptMTTKRP_3D+0x19c>
    3b68:	34000436 	cbz	w22, 3bec <sptMTTKRP_3D+0x19c>
		tmp_i_1 = times_inds_1[x];
		tmp_i_2 = times_inds_2[x];
		entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3b6c:	f9403fe4 	ldr	x4, [sp, #120]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3b70:	d2800009 	mov	x9, #0x0                   	// #0
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3b74:	f9400b47 	ldr	x7, [x26, #16]
    3b78:	f9400888 	ldr	x8, [x4, #16]
    3b7c:	d503201f 	nop
    3b80:	b8697b04 	ldr	w4, [x24, x9, lsl #2]
    3b84:	b8697b26 	ldr	w6, [x25, x9, lsl #2]
    3b88:	b8697b65 	ldr	w5, [x27, x9, lsl #2]
    3b8c:	1b047e84 	mul	w4, w20, w4
		entry = vals[x];
    3b90:	bc697ae3 	ldr	s3, [x23, x9, lsl #2]
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3b94:	1b067e86 	mul	w6, w20, w6
    3b98:	2a0403e0 	mov	w0, w4
    3b9c:	0b160084 	add	w4, w4, w22
    3ba0:	1b057e85 	mul	w5, w20, w5
    3ba4:	4b0000c6 	sub	w6, w6, w0
    3ba8:	4b0000a5 	sub	w5, w5, w0
    3bac:	d503201f 	nop
    3bb0:	0b0000c3 	add	w3, w6, w0
    3bb4:	0b0000a2 	add	w2, w5, w0
    3bb8:	2a0003e1 	mov	w1, w0
		for(sptIndex r=0; r<R; ++r) {
    3bbc:	11000400 	add	w0, w0, #0x1
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3bc0:	bc637901 	ldr	s1, [x8, x3, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3bc4:	6b04001f 	cmp	w0, w4
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3bc8:	bc6278e2 	ldr	s2, [x7, x2, lsl #2]
    3bcc:	bc617a60 	ldr	s0, [x19, x1, lsl #2]
    3bd0:	1e220821 	fmul	s1, s1, s2
    3bd4:	1f030020 	fmadd	s0, s1, s3, s0
    3bd8:	bc217a60 	str	s0, [x19, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3bdc:	54fffea1 	b.ne	3bb0 <sptMTTKRP_3D+0x160>  // b.any
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3be0:	91000529 	add	x9, x9, #0x1
    3be4:	eb0902bf 	cmp	x21, x9
    3be8:	54fffcc1 	b.ne	3b80 <sptMTTKRP_3D+0x130>  // b.any
		}
	}
	sptStopTimer(timer);
    3bec:	f94047e0 	ldr	x0, [sp, #136]
    3bf0:	940003a8 	bl	4a90 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    3bf4:	d0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    3bf8:	9127e021 	add	x1, x1, #0x9f8
    3bfc:	f94047e0 	ldr	x0, [sp, #136]
    3c00:	940003e4 	bl	4b90 <sptPrintElapsedTime>
    3c04:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3c08:	f94047e0 	ldr	x0, [sp, #136]
    3c0c:	94000421 	bl	4c90 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3c10:	1e604100 	fmov	d0, d8
    3c14:	d0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    3c18:	9128c000 	add	x0, x0, #0xa30
    3c1c:	97fff601 	bl	1420 <printf@plt>
	printf("\n");
    3c20:	52800140 	mov	w0, #0xa                   	// #10
    3c24:	97fff60b 	bl	1450 <putchar@plt>

	return 0;
    3c28:	fd4033e8 	ldr	d8, [sp, #96]
    3c2c:	52800000 	mov	w0, #0x0                   	// #0
}
    3c30:	a94153f3 	ldp	x19, x20, [sp, #16]
    3c34:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3c38:	a94363f7 	ldp	x23, x24, [sp, #48]
    3c3c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3c40:	a94573fb 	ldp	x27, x28, [sp, #80]
    3c44:	a8c97bfd 	ldp	x29, x30, [sp], #144
    3c48:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3c4c:	d0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    3c50:	d0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    3c54:	d0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    3c58:	9126a084 	add	x4, x4, #0x9a8
    3c5c:	9127e000 	add	x0, x0, #0x9f8
    3c60:	91274042 	add	x2, x2, #0x9d0
    3c64:	52801263 	mov	w3, #0x93                  	// #147
    3c68:	52800041 	mov	w1, #0x2                   	// #2
    3c6c:	94000489 	bl	4e90 <spt_ComplainError>
    3c70:	52800040 	mov	w0, #0x2                   	// #2
    3c74:	17ffffef 	b	3c30 <sptMTTKRP_3D+0x1e0>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3c78:	d0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    3c7c:	d0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    3c80:	d0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    3c84:	91284084 	add	x4, x4, #0xa10
    3c88:	9127e000 	add	x0, x0, #0x9f8
    3c8c:	91274042 	add	x2, x2, #0x9d0
    3c90:	528012c3 	mov	w3, #0x96                  	// #150
    3c94:	52800041 	mov	w1, #0x2                   	// #2
    3c98:	9400047e 	bl	4e90 <spt_ComplainError>
    3c9c:	52800040 	mov	w0, #0x2                   	// #2
    3ca0:	17ffffe4 	b	3c30 <sptMTTKRP_3D+0x1e0>
	sptAssert(nmodes ==3);
    3ca4:	12800000 	mov	w0, #0xffffffff            	// #-1
    3ca8:	fd0033e8 	str	d8, [sp, #96]
    3cac:	97fff54d 	bl	11e0 <exit@plt>

0000000000003cb0 <sptMTTKRP>:
							sptIndex const mode) {
    3cb0:	a9b47bfd 	stp	x29, x30, [sp, #-192]!
    3cb4:	910003fd 	mov	x29, sp
    3cb8:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const nmodes = X->nmodes;
    3cbc:	b9400015 	ldr	w21, [x0]
	if(nmodes == 3) {
    3cc0:	71000ebf 	cmp	w21, #0x3
    3cc4:	54002380 	b.eq	4134 <sptMTTKRP+0x484>  // b.none
	sptNnzIndex const nnz = X->nnz;
    3cc8:	a90363f7 	stp	x23, x24, [sp, #48]
    3ccc:	aa0003e5 	mov	x5, x0
	sptIndex const * const ndims = X->ndims;
    3cd0:	aa0103f6 	mov	x22, x1
    3cd4:	aa0203f8 	mov	x24, x2
    3cd8:	2a0303e6 	mov	w6, w3
	sptMatrix * const restrict M = mats[nmodes];
    3cdc:	f8755827 	ldr	x7, [x1, w21, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    3ce0:	f9400804 	ldr	x4, [x0, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    3ce4:	340003b5 	cbz	w21, 3d58 <sptMTTKRP+0xa8>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3ce8:	d2800000 	mov	x0, #0x0                   	// #0
    3cec:	b94004e3 	ldr	w3, [x7, #4]
    3cf0:	14000008 	b	3d10 <sptMTTKRP+0x60>
		if(mats[i]->nrows != ndims[i]) {
    3cf4:	b9400022 	ldr	w2, [x1]
    3cf8:	b8607881 	ldr	w1, [x4, x0, lsl #2]
    3cfc:	6b01005f 	cmp	w2, w1
    3d00:	54001fe1 	b.ne	40fc <sptMTTKRP+0x44c>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3d04:	91000400 	add	x0, x0, #0x1
    3d08:	6b0002bf 	cmp	w21, w0
    3d0c:	54000269 	b.ls	3d58 <sptMTTKRP+0xa8>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3d10:	f8607ac1 	ldr	x1, [x22, x0, lsl #3]
    3d14:	b9400422 	ldr	w2, [x1, #4]
    3d18:	6b03005f 	cmp	w2, w3
    3d1c:	54fffec0 	b.eq	3cf4 <sptMTTKRP+0x44>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3d20:	d0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    3d24:	d0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    3d28:	d0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    3d2c:	9126a084 	add	x4, x4, #0x9a8
    3d30:	9127e000 	add	x0, x0, #0x9f8
    3d34:	91274042 	add	x2, x2, #0x9d0
    3d38:	528007c3 	mov	w3, #0x3e                  	// #62
    3d3c:	52800041 	mov	w1, #0x2                   	// #2
    3d40:	94000454 	bl	4e90 <spt_ComplainError>
    3d44:	a94363f7 	ldp	x23, x24, [sp, #48]
    3d48:	52800040 	mov	w0, #0x2                   	// #2
}
    3d4c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3d50:	a8cc7bfd 	ldp	x29, x30, [sp], #192
    3d54:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    3d58:	f8665ac0 	ldr	x0, [x22, w6, uxtw #3]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d5c:	52800001 	mov	w1, #0x0                   	// #0
    3d60:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d64:	52800319 	mov	w25, #0x18                  	// #24
	sptIndex const stride = mats[0]->stride;
    3d68:	f94002c2 	ldr	x2, [x22]
    3d6c:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d70:	f94010a3 	ldr	x3, [x5, #32]
    3d74:	a90573fb 	stp	x27, x28, [sp, #80]
	sptNewValueVector(&scratch, R, R);
    3d78:	9102a3fc 	add	x28, sp, #0xa8
	sptIndex const stride = mats[0]->stride;
    3d7c:	b9400c57 	ldr	w23, [x2, #12]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d80:	9bb90cc3 	umaddl	x3, w6, w25, x3
	sptNewValueVector(&scratch, R, R);
    3d84:	f9004ffc 	str	x28, [sp, #152]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d88:	b9400002 	ldr	w2, [x0]
	sptValue * const restrict mvals = M->values;
    3d8c:	f94008f4 	ldr	x20, [x7, #16]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d90:	f940087b 	ldr	x27, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d94:	1b027ee2 	mul	w2, w23, w2
	sptNnzIndex const nnz = X->nnz;
    3d98:	f9400ca3 	ldr	x3, [x5, #24]
	sptIndex const R = mats[mode]->ncols;
    3d9c:	b9400413 	ldr	w19, [x0, #4]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3da0:	d37ef442 	lsl	x2, x2, #2
    3da4:	aa1403e0 	mov	x0, x20
	sptValue const * const restrict vals = X->values.data;
    3da8:	f9401cba 	ldr	x26, [x5, #56]
    3dac:	fd0033e8 	str	d8, [sp, #96]
	sptNnzIndex const nnz = X->nnz;
    3db0:	a9070fe5 	stp	x5, x3, [sp, #112]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3db4:	97fff547 	bl	12d0 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    3db8:	2a1303e2 	mov	w2, w19
    3dbc:	aa1c03e0 	mov	x0, x28
    3dc0:	aa0203e1 	mov	x1, x2
    3dc4:	97fff8c3 	bl	20d0 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    3dc8:	0f000400 	movi	v0.2s, #0x0
    3dcc:	aa1c03e0 	mov	x0, x28
    3dd0:	97fff8e8 	bl	2170 <sptConstantValueVector>
	sptNewTimer(&timer, 0);
    3dd4:	52800001 	mov	w1, #0x0                   	// #0
    3dd8:	910283e0 	add	x0, sp, #0xa0
    3ddc:	940002dd 	bl	4950 <sptNewTimer>
	sptStartTimer(timer);
    3de0:	f94053e0 	ldr	x0, [sp, #160]
    3de4:	940002fb 	bl	49d0 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3de8:	a9470fe5 	ldp	x5, x3, [sp, #112]
    3dec:	b4001203 	cbz	x3, 402c <sptMTTKRP+0x37c>
    3df0:	53027e60 	lsr	w0, w19, #2
    3df4:	121e766b 	and	w11, w19, #0xfffffffc
    3df8:	51000400 	sub	w0, w0, #0x1
			scratch.data[r] = entry * times_mat_values[r];
    3dfc:	11000566 	add	w6, w11, #0x1
		sptIndex times_mat_index = mats_order[1];
    3e00:	b9400701 	ldr	w1, [x24, #4]
		for(sptIndex r=0; r<R; ++r) {
    3e04:	11000964 	add	w4, w11, #0x2
			scratch.data[r] = entry * times_mat_values[r];
    3e08:	927e7672 	and	x18, x19, #0xfffffffc
    3e0c:	d37ef470 	lsl	x16, x3, #2
		for(sptIndex r=0; r<R; ++r) {
    3e10:	aa0403fc 	mov	x28, x4
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3e14:	f94010ad 	ldr	x13, [x5, #32]
    3e18:	91000405 	add	x5, x0, #0x1
		for(sptIndex r=0; r<R; ++r) {
    3e1c:	1100057e 	add	w30, w11, #0x1
    3e20:	4b0b026c 	sub	w12, w19, w11
    3e24:	5100066e 	sub	w14, w19, #0x1
			scratch.data[r] = entry * times_mat_values[r];
    3e28:	f9405fe0 	ldr	x0, [sp, #184]
    3e2c:	d280000a 	mov	x10, #0x0                   	// #0
    3e30:	d37ceca5 	lsl	x5, x5, #4
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3e34:	f8615ac2 	ldr	x2, [x22, w1, uxtw #3]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3e38:	9bb93421 	umaddl	x1, w1, w25, x13
			scratch.data[r] = entry * times_mat_values[r];
    3e3c:	8b040803 	add	x3, x0, x4, lsl #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3e40:	f9400831 	ldr	x17, [x1, #16]
			scratch.data[r] = entry * times_mat_values[r];
    3e44:	8b060801 	add	x1, x0, x6, lsl #2
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3e48:	f940084f 	ldr	x15, [x2, #16]
			scratch.data[r] = entry * times_mat_values[r];
    3e4c:	a90787e6 	stp	x6, x1, [sp, #120]
    3e50:	8b120801 	add	x1, x0, x18, lsl #2
    3e54:	f9003be1 	str	x1, [sp, #112]
    3e58:	a9088fe4 	stp	x4, x3, [sp, #136]
    3e5c:	d503201f 	nop
		sptIndex tmp_mult = tmp_i * stride;
    3e60:	b86a6a23 	ldr	w3, [x17, x10]
		sptValue const entry = vals[x];
    3e64:	bc6a6b42 	ldr	s2, [x26, x10]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3e68:	1b037ee3 	mul	w3, w23, w3
		for(sptIndex r=0; r<R; ++r) {
    3e6c:	34000db3 	cbz	w19, 4020 <sptMTTKRP+0x370>
    3e70:	91000461 	add	x1, x3, #0x1
    3e74:	d37ef421 	lsl	x1, x1, #2
    3e78:	8b0101e2 	add	x2, x15, x1
    3e7c:	cb020002 	sub	x2, x0, x2
    3e80:	f100205f 	cmp	x2, #0x8
    3e84:	7a4389c0 	ccmp	w14, #0x3, #0x0, hi  // hi = pmore
    3e88:	54001269 	b.ls	40d4 <sptMTTKRP+0x424>  // b.plast
    3e8c:	4e040441 	dup	v1.4s, v2.s[0]
    3e90:	d1001022 	sub	x2, x1, #0x4
    3e94:	d2800001 	mov	x1, #0x0                   	// #0
    3e98:	8b0201e2 	add	x2, x15, x2
    3e9c:	d503201f 	nop
			scratch.data[r] = entry * times_mat_values[r];
    3ea0:	3ce16840 	ldr	q0, [x2, x1]
    3ea4:	6e21dc00 	fmul	v0.4s, v0.4s, v1.4s
    3ea8:	3ca16800 	str	q0, [x0, x1]
		for(sptIndex r=0; r<R; ++r) {
    3eac:	91004021 	add	x1, x1, #0x10
    3eb0:	eb0100bf 	cmp	x5, x1
    3eb4:	54ffff61 	b.ne	3ea0 <sptMTTKRP+0x1f0>  // b.any
    3eb8:	6b0b027f 	cmp	w19, w11
    3ebc:	540002c0 	b.eq	3f14 <sptMTTKRP+0x264>  // b.none
			scratch.data[r] = entry * times_mat_values[r];
    3ec0:	8b120061 	add	x1, x3, x18
		for(sptIndex r=0; r<R; ++r) {
    3ec4:	6b1e027f 	cmp	w19, w30
			scratch.data[r] = entry * times_mat_values[r];
    3ec8:	bc6179e0 	ldr	s0, [x15, x1, lsl #2]
    3ecc:	f9403be1 	ldr	x1, [sp, #112]
    3ed0:	1e220800 	fmul	s0, s0, s2
    3ed4:	bd000020 	str	s0, [x1]
		for(sptIndex r=0; r<R; ++r) {
    3ed8:	540001e9 	b.ls	3f14 <sptMTTKRP+0x264>  // b.plast
			scratch.data[r] = entry * times_mat_values[r];
    3edc:	f9403fe1 	ldr	x1, [sp, #120]
		for(sptIndex r=0; r<R; ++r) {
    3ee0:	6b1c027f 	cmp	w19, w28
			scratch.data[r] = entry * times_mat_values[r];
    3ee4:	8b010061 	add	x1, x3, x1
    3ee8:	bc6179e0 	ldr	s0, [x15, x1, lsl #2]
    3eec:	f94043e1 	ldr	x1, [sp, #128]
    3ef0:	1e220800 	fmul	s0, s0, s2
    3ef4:	bd000020 	str	s0, [x1]
		for(sptIndex r=0; r<R; ++r) {
    3ef8:	540000e9 	b.ls	3f14 <sptMTTKRP+0x264>  // b.plast
			scratch.data[r] = entry * times_mat_values[r];
    3efc:	f94047e1 	ldr	x1, [sp, #136]
    3f00:	8b030023 	add	x3, x1, x3
    3f04:	f9404be1 	ldr	x1, [sp, #144]
    3f08:	bc6379e0 	ldr	s0, [x15, x3, lsl #2]
    3f0c:	1e220802 	fmul	s2, s0, s2
    3f10:	bd000022 	str	s2, [x1]
		for(sptIndex i=2; i<nmodes; ++i) {
    3f14:	71000abf 	cmp	w21, #0x2
    3f18:	d2800047 	mov	x7, #0x2                   	// #2
    3f1c:	540006c9 	b.ls	3ff4 <sptMTTKRP+0x344>  // b.plast
			times_mat_index = mats_order[i];
    3f20:	b8677b01 	ldr	w1, [x24, x7, lsl #2]
			tmp_i = times_inds[x];
    3f24:	9bb93423 	umaddl	x3, w1, w25, x13
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3f28:	f8615ac2 	ldr	x2, [x22, w1, uxtw #3]
			tmp_i = times_inds[x];
    3f2c:	f9400861 	ldr	x1, [x3, #16]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3f30:	f9400844 	ldr	x4, [x2, #16]
			tmp_mult = tmp_i * stride;
    3f34:	b86a6822 	ldr	w2, [x1, x10]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3f38:	1b027ee2 	mul	w2, w23, w2
			for(sptIndex r=0; r<R; ++r) {
    3f3c:	91000443 	add	x3, x2, #0x1
    3f40:	d37ef463 	lsl	x3, x3, #2
    3f44:	8b030081 	add	x1, x4, x3
    3f48:	cb010001 	sub	x1, x0, x1
    3f4c:	f100203f 	cmp	x1, #0x8
    3f50:	7a418a64 	ccmp	w19, #0x1, #0x4, hi  // hi = pmore
    3f54:	54000a00 	b.eq	4094 <sptMTTKRP+0x3e4>  // b.none
    3f58:	710009df 	cmp	w14, #0x2
    3f5c:	54000b29 	b.ls	40c0 <sptMTTKRP+0x410>  // b.plast
    3f60:	d1001063 	sub	x3, x3, #0x4
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3f64:	d2800001 	mov	x1, #0x0                   	// #0
    3f68:	8b030083 	add	x3, x4, x3
    3f6c:	d503201f 	nop
				scratch_data[r] *= times_mat_values[r];
    3f70:	3ce16860 	ldr	q0, [x3, x1]
    3f74:	3ce16801 	ldr	q1, [x0, x1]
    3f78:	6e21dc00 	fmul	v0.4s, v0.4s, v1.4s
    3f7c:	3ca16800 	str	q0, [x0, x1]
			for(sptIndex r=0; r<R; ++r) {
    3f80:	91004021 	add	x1, x1, #0x10
    3f84:	eb05003f 	cmp	x1, x5
    3f88:	54ffff41 	b.ne	3f70 <sptMTTKRP+0x2c0>  // b.any
    3f8c:	6b0b027f 	cmp	w19, w11
    3f90:	540002c0 	b.eq	3fe8 <sptMTTKRP+0x338>  // b.none
    3f94:	7100059f 	cmp	w12, #0x1
    3f98:	2a0c03e8 	mov	w8, w12
    3f9c:	54000980 	b.eq	40cc <sptMTTKRP+0x41c>  // b.none
    3fa0:	2a0b03e9 	mov	w9, w11
    3fa4:	2a0903e3 	mov	w3, w9
    3fa8:	121f7901 	and	w1, w8, #0xfffffffe
				scratch_data[r] *= times_mat_values[r];
    3fac:	8b020066 	add	x6, x3, x2
    3fb0:	d37ef463 	lsl	x3, x3, #2
    3fb4:	d37ef4c6 	lsl	x6, x6, #2
    3fb8:	6b08003f 	cmp	w1, w8
    3fbc:	fc636801 	ldr	d1, [x0, x3]
    3fc0:	0b090021 	add	w1, w1, w9
    3fc4:	fc666880 	ldr	d0, [x4, x6]
    3fc8:	2e21dc00 	fmul	v0.2s, v0.2s, v1.2s
    3fcc:	fc236800 	str	d0, [x0, x3]
			for(sptIndex r=0; r<R; ++r) {
    3fd0:	540000c0 	b.eq	3fe8 <sptMTTKRP+0x338>  // b.none
				scratch_data[r] *= times_mat_values[r];
    3fd4:	8b020022 	add	x2, x1, x2
    3fd8:	bc617800 	ldr	s0, [x0, x1, lsl #2]
    3fdc:	bc627881 	ldr	s1, [x4, x2, lsl #2]
    3fe0:	1e210800 	fmul	s0, s0, s1
    3fe4:	bc217800 	str	s0, [x0, x1, lsl #2]
		for(sptIndex i=2; i<nmodes; ++i) {
    3fe8:	910004e7 	add	x7, x7, #0x1
    3fec:	6b0702bf 	cmp	w21, w7
    3ff0:	54fff988 	b.hi	3f20 <sptMTTKRP+0x270>  // b.pmore
		sptIndex tmp_mode = mode_i * stride;
    3ff4:	b86a6b63 	ldr	w3, [x27, x10]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3ff8:	d2800001 	mov	x1, #0x0                   	// #0
		sptIndex tmp_mode = mode_i * stride;
    3ffc:	1b037ee3 	mul	w3, w23, w3
			mvals[tmp_mode + r] += scratch_data[r];
    4000:	0b010062 	add	w2, w3, w1
    4004:	bc617801 	ldr	s1, [x0, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    4008:	91000421 	add	x1, x1, #0x1
    400c:	6b01027f 	cmp	w19, w1
			mvals[tmp_mode + r] += scratch_data[r];
    4010:	bc627a80 	ldr	s0, [x20, x2, lsl #2]
    4014:	1e212800 	fadd	s0, s0, s1
    4018:	bc227a80 	str	s0, [x20, x2, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    401c:	54ffff28 	b.hi	4000 <sptMTTKRP+0x350>  // b.pmore
	for(sptNnzIndex x=0; x<nnz; ++x) {
    4020:	9100114a 	add	x10, x10, #0x4
    4024:	eb10015f 	cmp	x10, x16
    4028:	54fff1c1 	b.ne	3e60 <sptMTTKRP+0x1b0>  // b.any
	sptStopTimer(timer);
    402c:	f94053e0 	ldr	x0, [sp, #160]
    4030:	94000298 	bl	4a90 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    4034:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4038:	9127e021 	add	x1, x1, #0x9f8
    403c:	f94053e0 	ldr	x0, [sp, #160]
    4040:	940002d4 	bl	4b90 <sptPrintElapsedTime>
    4044:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    4048:	f94053e0 	ldr	x0, [sp, #160]
    404c:	94000311 	bl	4c90 <sptFreeTimer>
	sptFreeValueVector(&scratch);
    4050:	f9404fe0 	ldr	x0, [sp, #152]
    4054:	97fff8f3 	bl	2420 <sptFreeValueVector>
	printf("[Total time]: %lf\n", total_time);
    4058:	1e604100 	fmov	d0, d8
    405c:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4060:	9128c000 	add	x0, x0, #0xa30
    4064:	97fff4ef 	bl	1420 <printf@plt>
	printf("\n");
    4068:	52800140 	mov	w0, #0xa                   	// #10
    406c:	97fff4f9 	bl	1450 <putchar@plt>
	return 0;
    4070:	52800000 	mov	w0, #0x0                   	// #0
    4074:	a94153f3 	ldp	x19, x20, [sp, #16]
}
    4078:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    407c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4080:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4084:	a94573fb 	ldp	x27, x28, [sp, #80]
    4088:	fd4033e8 	ldr	d8, [sp, #96]
}
    408c:	a8cc7bfd 	ldp	x29, x30, [sp], #192
    4090:	d65f03c0 	ret
    4094:	8b020882 	add	x2, x4, x2, lsl #2
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    4098:	d2800001 	mov	x1, #0x0                   	// #0
    409c:	d503201f 	nop
				scratch_data[r] *= times_mat_values[r];
    40a0:	bc617800 	ldr	s0, [x0, x1, lsl #2]
    40a4:	bc617841 	ldr	s1, [x2, x1, lsl #2]
    40a8:	1e210800 	fmul	s0, s0, s1
    40ac:	bc217800 	str	s0, [x0, x1, lsl #2]
			for(sptIndex r=0; r<R; ++r) {
    40b0:	91000421 	add	x1, x1, #0x1
    40b4:	6b01027f 	cmp	w19, w1
    40b8:	54ffff48 	b.hi	40a0 <sptMTTKRP+0x3f0>  // b.pmore
    40bc:	17ffffcb 	b	3fe8 <sptMTTKRP+0x338>
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    40c0:	2a1303e8 	mov	w8, w19
			for(sptIndex r=0; r<R; ++r) {
    40c4:	52800009 	mov	w9, #0x0                   	// #0
    40c8:	17ffffb7 	b	3fa4 <sptMTTKRP+0x2f4>
    40cc:	2a0b03e1 	mov	w1, w11
    40d0:	17ffffc1 	b	3fd4 <sptMTTKRP+0x324>
    40d4:	8b0309e3 	add	x3, x15, x3, lsl #2
		for(sptIndex r=0; r<R; ++r) {
    40d8:	d2800001 	mov	x1, #0x0                   	// #0
    40dc:	d503201f 	nop
			scratch.data[r] = entry * times_mat_values[r];
    40e0:	bc617860 	ldr	s0, [x3, x1, lsl #2]
    40e4:	1e220800 	fmul	s0, s0, s2
    40e8:	bc217800 	str	s0, [x0, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    40ec:	91000421 	add	x1, x1, #0x1
    40f0:	6b01027f 	cmp	w19, w1
    40f4:	54ffff68 	b.hi	40e0 <sptMTTKRP+0x430>  // b.pmore
    40f8:	17ffff87 	b	3f14 <sptMTTKRP+0x264>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    40fc:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    4100:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4104:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4108:	91284084 	add	x4, x4, #0xa10
    410c:	91274042 	add	x2, x2, #0x9d0
    4110:	9127e000 	add	x0, x0, #0x9f8
    4114:	52800823 	mov	w3, #0x41                  	// #65
    4118:	52800041 	mov	w1, #0x2                   	// #2
    411c:	9400035d 	bl	4e90 <spt_ComplainError>
    4120:	52800040 	mov	w0, #0x2                   	// #2
}
    4124:	a9425bf5 	ldp	x21, x22, [sp, #32]
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4128:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    412c:	a8cc7bfd 	ldp	x29, x30, [sp], #192
    4130:	d65f03c0 	ret
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    4134:	97fffe47 	bl	3a50 <sptMTTKRP_3D>
    4138:	34ffe0a0 	cbz	w0, 3d4c <sptMTTKRP+0x9c>
    413c:	12800000 	mov	w0, #0xffffffff            	// #-1
    4140:	a90153f3 	stp	x19, x20, [sp, #16]
    4144:	a90363f7 	stp	x23, x24, [sp, #48]
    4148:	a9046bf9 	stp	x25, x26, [sp, #64]
    414c:	a90573fb 	stp	x27, x28, [sp, #80]
    4150:	fd0033e8 	str	d8, [sp, #96]
    4154:	97fff423 	bl	11e0 <exit@plt>
    4158:	d503201f 	nop
    415c:	d503201f 	nop

0000000000004160 <sptOmpMTTKRP._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    4160:	a9b27bfd 	stp	x29, x30, [sp, #-224]!
    4164:	910003fd 	mov	x29, sp
    4168:	a9025bf5 	stp	x21, x22, [sp, #32]
    416c:	f9400c15 	ldr	x21, [x0, #24]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    4170:	b40010b5 	cbz	x21, 4384 <sptOmpMTTKRP._omp_fn.0+0x224>
    4174:	a90153f3 	stp	x19, x20, [sp, #16]
    4178:	aa0003f4 	mov	x20, x0
    417c:	97fff469 	bl	1320 <omp_get_num_threads@plt>
    4180:	93407c13 	sxtw	x19, w0
    4184:	97fff42f 	bl	1240 <omp_get_thread_num@plt>
    4188:	9ad30aa1 	udiv	x1, x21, x19
    418c:	93407c16 	sxtw	x22, w0
    4190:	9b13d420 	msub	x0, x1, x19, x21
    4194:	eb0002df 	cmp	x22, x0
    4198:	54001043 	b.cc	43a0 <sptOmpMTTKRP._omp_fn.0+0x240>  // b.lo, b.ul, b.last
    419c:	9b160036 	madd	x22, x1, x22, x0
    41a0:	8b160020 	add	x0, x1, x22
    41a4:	eb0002df 	cmp	x22, x0
    41a8:	f9004be0 	str	x0, [sp, #144]
    41ac:	54000f22 	b.cs	4390 <sptOmpMTTKRP._omp_fn.0+0x230>  // b.hs, b.nlast
    41b0:	910323e0 	add	x0, sp, #0xc8
    41b4:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    41b8:	b940429c 	ldr	w28, [x20, #64]
    41bc:	f90047e0 	str	x0, [sp, #136]
    41c0:	f9400280 	ldr	x0, [x20]
    41c4:	a90363f7 	stp	x23, x24, [sp, #48]
    41c8:	51000795 	sub	w21, w28, #0x1
    41cc:	f9400698 	ldr	x24, [x20, #8]
    41d0:	910006b5 	add	x21, x21, #0x1
    41d4:	f90053e0 	str	x0, [sp, #160]
		sptValueVector scratch;  // Temporary array
		sptNewValueVector(&scratch, R, R);
    41d8:	2a1c03e0 	mov	w0, w28
    41dc:	d37ef6b5 	lsl	x21, x21, #2
    41e0:	a9046bf9 	stp	x25, x26, [sp, #64]
		sptConstantValueVector(&scratch, 0);

		sptIndex times_mat_index = mats_order[1];
		sptMatrix * times_mat = mats[times_mat_index];
		sptIndex * times_inds = X->inds[times_mat_index].data;
    41e4:	52800319 	mov	w25, #0x18                  	// #24
		sptNewValueVector(&scratch, R, R);
    41e8:	f90057e0 	str	x0, [sp, #168]
#pragma omp parallel for schedule(static) num_threads(tk)
    41ec:	f9400a80 	ldr	x0, [x20, #16]
    41f0:	fd0033e8 	str	d8, [sp, #96]
    41f4:	f9003fe0 	str	x0, [sp, #120]
    41f8:	f9401280 	ldr	x0, [x20, #32]
    41fc:	f9004fe0 	str	x0, [sp, #152]
    4200:	f9401680 	ldr	x0, [x20, #40]
    4204:	f9005fe0 	str	x0, [sp, #184]
    4208:	f9401a80 	ldr	x0, [x20, #48]
    420c:	f9005be0 	str	x0, [sp, #176]
    4210:	29475280 	ldp	w0, w20, [x20, #56]
    4214:	b90087e0 	str	w0, [sp, #132]
		sptNewValueVector(&scratch, R, R);
    4218:	f94047f3 	ldr	x19, [sp, #136]
    421c:	f94057e1 	ldr	x1, [sp, #168]
    4220:	aa1303e0 	mov	x0, x19
    4224:	aa0103e2 	mov	x2, x1
    4228:	97fff7aa 	bl	20d0 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    422c:	0f000400 	movi	v0.2s, #0x0
    4230:	aa1303e0 	mov	x0, x19
    4234:	97fff7cf 	bl	2170 <sptConstantValueVector>
		sptIndex times_mat_index = mats_order[1];
    4238:	f9403fe0 	ldr	x0, [sp, #120]
    423c:	d37ef6c8 	lsl	x8, x22, #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    4240:	f94053e1 	ldr	x1, [sp, #160]
		sptIndex times_mat_index = mats_order[1];
    4244:	b9400400 	ldr	w0, [x0, #4]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    4248:	f9401027 	ldr	x7, [x1, #32]
		sptIndex tmp_i = times_inds[x];
		sptValue const entry = vals[x];
    424c:	f9404fe1 	ldr	x1, [sp, #152]
		sptMatrix * times_mat = mats[times_mat_index];
    4250:	f8605b02 	ldr	x2, [x24, w0, uxtw #3]
		sptIndex tmp_i = times_inds[x];
    4254:	9bb91c00 	umaddl	x0, w0, w25, x7
		sptValue const entry = vals[x];
    4258:	bc767821 	ldr	s1, [x1, x22, lsl #2]
		sptIndex tmp_i = times_inds[x];
    425c:	f9400800 	ldr	x0, [x0, #16]
    4260:	b8767803 	ldr	w3, [x0, x22, lsl #2]
#pragma omp simd
		for(sptIndex r=0; r<R; ++r) {
    4264:	340007bc 	cbz	w28, 4358 <sptOmpMTTKRP._omp_fn.0+0x1f8>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    4268:	1b037e83 	mul	w3, w20, w3
    426c:	d2800000 	mov	x0, #0x0                   	// #0
    4270:	f9400845 	ldr	x5, [x2, #16]
    4274:	f9406fe1 	ldr	x1, [sp, #216]
    4278:	0b000062 	add	w2, w3, w0
    427c:	bc6278a0 	ldr	s0, [x5, x2, lsl #2]
    4280:	1e210800 	fmul	s0, s0, s1
    4284:	bc207820 	str	s0, [x1, x0, lsl #2]
    4288:	91000400 	add	x0, x0, #0x1
    428c:	6b00039f 	cmp	w28, w0
    4290:	54ffff48 	b.hi	4278 <sptOmpMTTKRP._omp_fn.0+0x118>  // b.pmore
		}

		for(sptIndex i=2; i<nmodes; ++i) {
    4294:	b94087e0 	ldr	w0, [sp, #132]
			times_inds = X->inds[times_mat_index].data;
			tmp_i = times_inds[x];

#pragma omp simd
			for(sptIndex r=0; r<R; ++r) {
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    4298:	d2800046 	mov	x6, #0x2                   	// #2
		for(sptIndex i=2; i<nmodes; ++i) {
    429c:	7100081f 	cmp	w0, #0x2
    42a0:	54000309 	b.ls	4300 <sptOmpMTTKRP._omp_fn.0+0x1a0>  // b.plast
    42a4:	d503201f 	nop
			times_mat_index = mats_order[i];
    42a8:	f9403fe0 	ldr	x0, [sp, #120]
    42ac:	b8667802 	ldr	w2, [x0, x6, lsl #2]
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    42b0:	d2800000 	mov	x0, #0x0                   	// #0
			tmp_i = times_inds[x];
    42b4:	9bb91c45 	umaddl	x5, w2, w25, x7
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    42b8:	f8625b03 	ldr	x3, [x24, w2, uxtw #3]
			tmp_i = times_inds[x];
    42bc:	f94008a2 	ldr	x2, [x5, #16]
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    42c0:	f9400865 	ldr	x5, [x3, #16]
    42c4:	b8686843 	ldr	w3, [x2, x8]
    42c8:	1b037e83 	mul	w3, w20, w3
    42cc:	d503201f 	nop
    42d0:	0b000062 	add	w2, w3, w0
    42d4:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    42d8:	bc6278a1 	ldr	s1, [x5, x2, lsl #2]
    42dc:	1e210800 	fmul	s0, s0, s1
    42e0:	bc207820 	str	s0, [x1, x0, lsl #2]
    42e4:	91000400 	add	x0, x0, #0x1
    42e8:	6b00039f 	cmp	w28, w0
    42ec:	54ffff28 	b.hi	42d0 <sptOmpMTTKRP._omp_fn.0+0x170>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    42f0:	b94087e0 	ldr	w0, [sp, #132]
    42f4:	910004c6 	add	x6, x6, #0x1
    42f8:	6b06001f 	cmp	w0, w6
    42fc:	54fffd68 	b.hi	42a8 <sptOmpMTTKRP._omp_fn.0+0x148>  // b.pmore
			}
		}

		sptIndex const mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    4300:	f9405fe0 	ldr	x0, [sp, #184]
#pragma omp parallel for schedule(static) num_threads(tk)
    4304:	d280001b 	mov	x27, #0x0                   	// #0
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    4308:	b8767813 	ldr	w19, [x0, x22, lsl #2]
    430c:	f9405be0 	ldr	x0, [sp, #176]
    4310:	1b137e93 	mul	w19, w20, w19
		for(sptIndex r=0; r<R; ++r) {
    4314:	8b130813 	add	x19, x0, x19, lsl #2
    4318:	14000002 	b	4320 <sptOmpMTTKRP._omp_fn.0+0x1c0>
    431c:	f9406fe1 	ldr	x1, [sp, #216]
    4320:	8b130377 	add	x23, x27, x19
#pragma omp atomic update
			mvals_row[r] += scratch.data[r];
    4324:	bc7b6828 	ldr	s8, [x1, x27]
    4328:	b94002fa 	ldr	w26, [x23]
    432c:	1e270340 	fmov	s0, w26
    4330:	aa1703e2 	mov	x2, x23
    4334:	2a1a03e0 	mov	w0, w26
    4338:	1e202900 	fadd	s0, s8, s0
    433c:	1e260001 	fmov	w1, s0
    4340:	94000370 	bl	5100 <__aarch64_cas4_acq_rel>
    4344:	6b00035f 	cmp	w26, w0
    4348:	54000321 	b.ne	43ac <sptOmpMTTKRP._omp_fn.0+0x24c>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    434c:	9100137b 	add	x27, x27, #0x4
    4350:	eb15037f 	cmp	x27, x21
    4354:	54fffe41 	b.ne	431c <sptOmpMTTKRP._omp_fn.0+0x1bc>  // b.any
		}

		sptFreeValueVector(&scratch);
    4358:	f94047e0 	ldr	x0, [sp, #136]
    435c:	910006d6 	add	x22, x22, #0x1
    4360:	97fff830 	bl	2420 <sptFreeValueVector>
    4364:	f9404be0 	ldr	x0, [sp, #144]
    4368:	eb16001f 	cmp	x0, x22
    436c:	54fff568 	b.hi	4218 <sptOmpMTTKRP._omp_fn.0+0xb8>  // b.pmore
    4370:	fd4033e8 	ldr	d8, [sp, #96]
    4374:	a94153f3 	ldp	x19, x20, [sp, #16]
    4378:	a94363f7 	ldp	x23, x24, [sp, #48]
    437c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4380:	a94573fb 	ldp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    4384:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4388:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    438c:	d65f03c0 	ret
    4390:	a94153f3 	ldp	x19, x20, [sp, #16]
    4394:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4398:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    439c:	d65f03c0 	ret
    43a0:	91000421 	add	x1, x1, #0x1
    43a4:	d2800000 	mov	x0, #0x0                   	// #0
    43a8:	17ffff7d 	b	419c <sptOmpMTTKRP._omp_fn.0+0x3c>
			mvals_row[r] += scratch.data[r];
    43ac:	2a0003fa 	mov	w26, w0
    43b0:	17ffffdf 	b	432c <sptOmpMTTKRP._omp_fn.0+0x1cc>
    43b4:	d503201f 	nop
    43b8:	d503201f 	nop
    43bc:	d503201f 	nop

00000000000043c0 <sptOmpMTTKRP_3D._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    43c0:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    43c4:	910003fd 	mov	x29, sp
    43c8:	a9025bf5 	stp	x21, x22, [sp, #32]
    43cc:	f9400016 	ldr	x22, [x0]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    43d0:	b40009f6 	cbz	x22, 450c <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
    43d4:	a90153f3 	stp	x19, x20, [sp, #16]
    43d8:	aa0003f3 	mov	x19, x0
    43dc:	a90363f7 	stp	x23, x24, [sp, #48]
    43e0:	97fff3d0 	bl	1320 <omp_get_num_threads@plt>
    43e4:	93407c14 	sxtw	x20, w0
    43e8:	97fff396 	bl	1240 <omp_get_thread_num@plt>
    43ec:	9ad40ad7 	udiv	x23, x22, x20
    43f0:	93407c15 	sxtw	x21, w0
    43f4:	9b14dae0 	msub	x0, x23, x20, x22
    43f8:	eb0002bf 	cmp	x21, x0
    43fc:	54000983 	b.cc	452c <sptOmpMTTKRP_3D._omp_fn.0+0x16c>  // b.lo, b.ul, b.last
    4400:	9b1502f5 	madd	x21, x23, x21, x0
    4404:	8b1502e0 	add	x0, x23, x21
    4408:	eb0002bf 	cmp	x21, x0
    440c:	f9003be0 	str	x0, [sp, #112]
    4410:	54000842 	b.cs	4518 <sptOmpMTTKRP_3D._omp_fn.0+0x158>  // b.hs, b.nlast
    4414:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    4418:	a940ee61 	ldp	x1, x27, [x19, #8]
    441c:	29480276 	ldp	w22, w0, [x19, #64]
    4420:	f9004fe1 	str	x1, [sp, #152]
    4424:	f9400e61 	ldr	x1, [x19, #24]
    4428:	f9401278 	ldr	x24, [x19, #32]
    442c:	b90097e0 	str	w0, [sp, #148]
    4430:	f9401a7c 	ldr	x28, [x19, #48]
    4434:	f90047e1 	str	x1, [sp, #136]
    4438:	f9401661 	ldr	x1, [x19, #40]
    443c:	f90043e1 	str	x1, [sp, #128]
    4440:	f9401e61 	ldr	x1, [x19, #56]
    4444:	f9003fe1 	str	x1, [sp, #120]
    4448:	34000780 	cbz	w0, 4538 <sptOmpMTTKRP_3D._omp_fn.0+0x178>
    444c:	a9046bf9 	stp	x25, x26, [sp, #64]
    4450:	6d0627e8 	stp	d8, d9, [sp, #96]
    4454:	d503201f 	nop
		sptIndex tmp_i_2 = times_inds_2[x];
		sptValue entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4458:	f9403fe0 	ldr	x0, [sp, #120]
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    445c:	b8757b62 	ldr	w2, [x27, x21, lsl #2]
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4460:	b8757813 	ldr	w19, [x0, x21, lsl #2]
    4464:	f94043e0 	ldr	x0, [sp, #128]
    4468:	1b027ec2 	mul	w2, w22, w2
    446c:	1b137ed3 	mul	w19, w22, w19
    4470:	b8757814 	ldr	w20, [x0, x21, lsl #2]
    4474:	2a1303f9 	mov	w25, w19
		sptValue entry = vals[x];
    4478:	f9404fe0 	ldr	x0, [sp, #152]
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    447c:	1b147ed4 	mul	w20, w22, w20
		sptValue entry = vals[x];
    4480:	bc757809 	ldr	s9, [x0, x21, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    4484:	4b130294 	sub	w20, w20, w19
    4488:	b94097e0 	ldr	w0, [sp, #148]
    448c:	0b130013 	add	w19, w0, w19
    4490:	f94047e0 	ldr	x0, [sp, #136]
    4494:	8b020817 	add	x23, x0, x2, lsl #2
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4498:	f9400b04 	ldr	x4, [x24, #16]
    449c:	0b190280 	add	w0, w20, w25
    44a0:	f9400b81 	ldr	x1, [x28, #16]
    44a4:	bc607888 	ldr	s8, [x4, x0, lsl #2]
    44a8:	bc795820 	ldr	s0, [x1, w25, uxtw #2]
    44ac:	1e200908 	fmul	s8, s8, s0
    44b0:	1e290908 	fmul	s8, s8, s9
    44b4:	b94002fa 	ldr	w26, [x23]
    44b8:	1e270340 	fmov	s0, w26
    44bc:	aa1703e2 	mov	x2, x23
    44c0:	2a1a03e0 	mov	w0, w26
    44c4:	1e202900 	fadd	s0, s8, s0
    44c8:	1e260001 	fmov	w1, s0
    44cc:	9400030d 	bl	5100 <__aarch64_cas4_acq_rel>
    44d0:	6b00035f 	cmp	w26, w0
    44d4:	540003a1 	b.ne	4548 <sptOmpMTTKRP_3D._omp_fn.0+0x188>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    44d8:	11000739 	add	w25, w25, #0x1
    44dc:	910012f7 	add	x23, x23, #0x4
    44e0:	6b19027f 	cmp	w19, w25
    44e4:	54fffda1 	b.ne	4498 <sptOmpMTTKRP_3D._omp_fn.0+0xd8>  // b.any
    44e8:	f9403be0 	ldr	x0, [sp, #112]
    44ec:	910006b5 	add	x21, x21, #0x1
    44f0:	eb15001f 	cmp	x0, x21
    44f4:	54fffb21 	b.ne	4458 <sptOmpMTTKRP_3D._omp_fn.0+0x98>  // b.any
    44f8:	6d4627e8 	ldp	d8, d9, [sp, #96]
    44fc:	a94153f3 	ldp	x19, x20, [sp, #16]
    4500:	a94363f7 	ldp	x23, x24, [sp, #48]
    4504:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4508:	a94573fb 	ldp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    450c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4510:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4514:	d65f03c0 	ret
    4518:	a94153f3 	ldp	x19, x20, [sp, #16]
    451c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4520:	a94363f7 	ldp	x23, x24, [sp, #48]
    4524:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4528:	d65f03c0 	ret
    452c:	910006f7 	add	x23, x23, #0x1
    4530:	d2800000 	mov	x0, #0x0                   	// #0
    4534:	17ffffb3 	b	4400 <sptOmpMTTKRP_3D._omp_fn.0+0x40>
    4538:	a94153f3 	ldp	x19, x20, [sp, #16]
    453c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4540:	a94573fb 	ldp	x27, x28, [sp, #80]
    4544:	17fffff2 	b	450c <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4548:	2a0003fa 	mov	w26, w0
    454c:	17ffffdb 	b	44b8 <sptOmpMTTKRP_3D._omp_fn.0+0xf8>

0000000000004550 <sptOmpMTTKRP_3D>:
{
    4550:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    4554:	910003fd 	mov	x29, sp
    4558:	a90153f3 	stp	x19, x20, [sp, #16]
    455c:	aa0103f3 	mov	x19, x1
    4560:	aa0003f4 	mov	x20, x0
	sptAssert(nmodes ==3);
    4564:	b9400001 	ldr	w1, [x0]
{
    4568:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex const stride = mats[0]->stride;
    456c:	f9400260 	ldr	x0, [x19]
	sptAssert(nmodes ==3);
    4570:	71000c3f 	cmp	w1, #0x3
{
    4574:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptNnzIndex const nnz = X->nnz;
    4578:	a9416a85 	ldp	x5, x26, [x20, #16]
	sptIndex const stride = mats[0]->stride;
    457c:	b9400c18 	ldr	w24, [x0, #12]
	sptValue const * const restrict vals = X->values.data;
    4580:	f9401e99 	ldr	x25, [x20, #56]
	sptAssert(nmodes ==3);
    4584:	a9025bf5 	stp	x21, x22, [sp, #32]
    4588:	54000e41 	b.ne	4750 <sptOmpMTTKRP_3D+0x200>  // b.any
    458c:	aa0203f6 	mov	x22, x2
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4590:	f9400e66 	ldr	x6, [x19, #24]
    4594:	b9400402 	ldr	w2, [x0, #4]
    4598:	b94004c1 	ldr	w1, [x6, #4]
    459c:	6b02003f 	cmp	w1, w2
    45a0:	54000ac1 	b.ne	46f8 <sptOmpMTTKRP_3D+0x1a8>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    45a4:	b9400002 	ldr	w2, [x0]
    45a8:	b94000a0 	ldr	w0, [x5]
    45ac:	6b00005f 	cmp	w2, w0
    45b0:	54000ba1 	b.ne	4724 <sptOmpMTTKRP_3D+0x1d4>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    45b4:	f9400660 	ldr	x0, [x19, #8]
    45b8:	b9400402 	ldr	w2, [x0, #4]
    45bc:	6b02003f 	cmp	w1, w2
    45c0:	540009c1 	b.ne	46f8 <sptOmpMTTKRP_3D+0x1a8>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    45c4:	b9400002 	ldr	w2, [x0]
    45c8:	b94004a0 	ldr	w0, [x5, #4]
    45cc:	6b00005f 	cmp	w2, w0
    45d0:	54000aa1 	b.ne	4724 <sptOmpMTTKRP_3D+0x1d4>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    45d4:	f9400a60 	ldr	x0, [x19, #16]
    45d8:	b9400402 	ldr	w2, [x0, #4]
    45dc:	6b01005f 	cmp	w2, w1
    45e0:	540008c1 	b.ne	46f8 <sptOmpMTTKRP_3D+0x1a8>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    45e4:	b9400001 	ldr	w1, [x0]
    45e8:	b94008a0 	ldr	w0, [x5, #8]
    45ec:	6b00003f 	cmp	w1, w0
    45f0:	540009a1 	b.ne	4724 <sptOmpMTTKRP_3D+0x1d4>  // b.any
	sptIndex const tmpI = mats[mode]->nrows;
    45f4:	f8635a60 	ldr	x0, [x19, w3, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    45f8:	52800315 	mov	w21, #0x18                  	// #24
    45fc:	2a0403f7 	mov	w23, w4
    4600:	a90573fb 	stp	x27, x28, [sp, #80]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4604:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    4608:	f9401282 	ldr	x2, [x20, #32]
	sptIndex const R = mats[mode]->ncols;
    460c:	29407005 	ldp	w5, w28, [x0]
	sptValue * const restrict mvals = mats[nmodes]->values;
    4610:	f94008db 	ldr	x27, [x6, #16]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    4614:	9bb50863 	umaddl	x3, w3, w21, x2
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4618:	1b057f02 	mul	w2, w24, w5
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    461c:	f9400864 	ldr	x4, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4620:	aa1b03e0 	mov	x0, x27
    4624:	fd0033e8 	str	d8, [sp, #96]
    4628:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    462c:	f9003fe4 	str	x4, [sp, #120]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4630:	97fff328 	bl	12d0 <memset@plt>
	sptIndex times_mat_index_2 = mats_order[2];
    4634:	29408ec5 	ldp	w5, w3, [x22, #4]
	sptNewTimer(&timer, 0);
    4638:	52800001 	mov	w1, #0x0                   	// #0
    463c:	910203e0 	add	x0, sp, #0x80
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4640:	f9401286 	ldr	x6, [x20, #32]
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    4644:	f8655a76 	ldr	x22, [x19, w5, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4648:	9bb518a5 	umaddl	x5, w5, w21, x6
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    464c:	f8635a74 	ldr	x20, [x19, w3, uxtw #3]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    4650:	9bb51863 	umaddl	x3, w3, w21, x6
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4654:	f94008b5 	ldr	x21, [x5, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    4658:	f9400873 	ldr	x19, [x3, #16]
	sptNewTimer(&timer, 0);
    465c:	940000bd 	bl	4950 <sptNewTimer>
	sptStartTimer(timer);
    4660:	f94043e0 	ldr	x0, [sp, #128]
    4664:	940000db 	bl	49d0 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    4668:	f9403fe4 	ldr	x4, [sp, #120]
    466c:	90000000 	adrp	x0, 4000 <sptMTTKRP+0x350>
    4670:	2a1703e2 	mov	w2, w23
    4674:	52800003 	mov	w3, #0x0                   	// #0
    4678:	910223e1 	add	x1, sp, #0x88
    467c:	a908e7fa 	stp	x26, x25, [sp, #136]
    4680:	910f0000 	add	x0, x0, #0x3c0
    4684:	a90ad7f6 	stp	x22, x21, [sp, #168]
    4688:	a909efe4 	stp	x4, x27, [sp, #152]
    468c:	a90bcff4 	stp	x20, x19, [sp, #184]
    4690:	291973f8 	stp	w24, w28, [sp, #200]
    4694:	97fff36b 	bl	1440 <GOMP_parallel@plt>
		}
	}
	sptStopTimer(timer);
    4698:	f94043e0 	ldr	x0, [sp, #128]
    469c:	940000fd 	bl	4a90 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    46a0:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    46a4:	9129c021 	add	x1, x1, #0xa70
    46a8:	f94043e0 	ldr	x0, [sp, #128]
    46ac:	94000139 	bl	4b90 <sptPrintElapsedTime>
    46b0:	1e604008 	fmov	d8, d0

	sptFreeTimer(timer);
    46b4:	f94043e0 	ldr	x0, [sp, #128]
    46b8:	94000176 	bl	4c90 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    46bc:	1e604100 	fmov	d0, d8
    46c0:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    46c4:	9128c000 	add	x0, x0, #0xa30
    46c8:	97fff356 	bl	1420 <printf@plt>
	printf("\n");
    46cc:	52800140 	mov	w0, #0xa                   	// #10
    46d0:	97fff360 	bl	1450 <putchar@plt>

	return 0;
    46d4:	fd4033e8 	ldr	d8, [sp, #96]
    46d8:	52800000 	mov	w0, #0x0                   	// #0
    46dc:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    46e0:	a94153f3 	ldp	x19, x20, [sp, #16]
    46e4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    46e8:	a94363f7 	ldp	x23, x24, [sp, #48]
    46ec:	a9446bf9 	ldp	x25, x26, [sp, #64]
    46f0:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    46f4:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    46f8:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    46fc:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4700:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4704:	9126a084 	add	x4, x4, #0x9a8
    4708:	9129c000 	add	x0, x0, #0xa70
    470c:	91292042 	add	x2, x2, #0xa48
    4710:	528012c3 	mov	w3, #0x96                  	// #150
    4714:	52800041 	mov	w1, #0x2                   	// #2
    4718:	940001de 	bl	4e90 <spt_ComplainError>
    471c:	52800040 	mov	w0, #0x2                   	// #2
    4720:	17fffff0 	b	46e0 <sptOmpMTTKRP_3D+0x190>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4724:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    4728:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    472c:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4730:	91284084 	add	x4, x4, #0xa10
    4734:	9129c000 	add	x0, x0, #0xa70
    4738:	91292042 	add	x2, x2, #0xa48
    473c:	52801323 	mov	w3, #0x99                  	// #153
    4740:	52800041 	mov	w1, #0x2                   	// #2
    4744:	940001d3 	bl	4e90 <spt_ComplainError>
    4748:	52800040 	mov	w0, #0x2                   	// #2
    474c:	17ffffe5 	b	46e0 <sptOmpMTTKRP_3D+0x190>
	sptAssert(nmodes ==3);
    4750:	12800000 	mov	w0, #0xffffffff            	// #-1
    4754:	a90573fb 	stp	x27, x28, [sp, #80]
    4758:	fd0033e8 	str	d8, [sp, #96]
    475c:	97fff2a1 	bl	11e0 <exit@plt>

0000000000004760 <sptOmpMTTKRP>:
{
    4760:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    4764:	910003fd 	mov	x29, sp
    4768:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const nmodes = X->nmodes;
    476c:	b9400016 	ldr	w22, [x0]
	if(nmodes == 3) {
    4770:	71000edf 	cmp	w22, #0x3
    4774:	54000dc0 	b.eq	492c <sptOmpMTTKRP+0x1cc>  // b.none
	sptNnzIndex const nnz = X->nnz;
    4778:	a90153f3 	stp	x19, x20, [sp, #16]
    477c:	aa0203f5 	mov	x21, x2
    4780:	aa0103f3 	mov	x19, x1
    4784:	2a0303e9 	mov	w9, w3
    4788:	2a0403f4 	mov	w20, w4
    478c:	a90363f7 	stp	x23, x24, [sp, #48]
    4790:	aa0003f7 	mov	x23, x0
	sptValue * const restrict mvals = mats[nmodes]->values;
    4794:	f8765820 	ldr	x0, [x1, w22, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    4798:	f9400ae8 	ldr	x8, [x23, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    479c:	340003d6 	cbz	w22, 4814 <sptOmpMTTKRP+0xb4>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    47a0:	d2800005 	mov	x5, #0x0                   	// #0
    47a4:	b9400401 	ldr	w1, [x0, #4]
    47a8:	14000008 	b	47c8 <sptOmpMTTKRP+0x68>
		if(mats[i]->nrows != ndims[i]) {
    47ac:	b94000c7 	ldr	w7, [x6]
    47b0:	b8657906 	ldr	w6, [x8, x5, lsl #2]
    47b4:	6b0600ff 	cmp	w7, w6
    47b8:	54000aa1 	b.ne	490c <sptOmpMTTKRP+0x1ac>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    47bc:	910004a5 	add	x5, x5, #0x1
    47c0:	6b0502df 	cmp	w22, w5
    47c4:	54000289 	b.ls	4814 <sptOmpMTTKRP+0xb4>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    47c8:	f8657a66 	ldr	x6, [x19, x5, lsl #3]
    47cc:	b94004c7 	ldr	w7, [x6, #4]
    47d0:	6b0100ff 	cmp	w7, w1
    47d4:	54fffec0 	b.eq	47ac <sptOmpMTTKRP+0x4c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    47d8:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    47dc:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    47e0:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    47e4:	9126a084 	add	x4, x4, #0x9a8
    47e8:	91292042 	add	x2, x2, #0xa48
    47ec:	9129c000 	add	x0, x0, #0xa70
    47f0:	52800823 	mov	w3, #0x41                  	// #65
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    47f4:	52800041 	mov	w1, #0x2                   	// #2
    47f8:	940001a6 	bl	4e90 <spt_ComplainError>
    47fc:	a94153f3 	ldp	x19, x20, [sp, #16]
    4800:	52800040 	mov	w0, #0x2                   	// #2
    4804:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    4808:	a9425bf5 	ldp	x21, x22, [sp, #32]
    480c:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4810:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    4814:	f8695a64 	ldr	x4, [x19, w9, uxtw #3]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4818:	52800305 	mov	w5, #0x18                  	// #24
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    481c:	52800001 	mov	w1, #0x0                   	// #0
    4820:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const stride = mats[0]->stride;
    4824:	f9400262 	ldr	x2, [x19]
    4828:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex const R = mats[mode]->ncols;
    482c:	b940049a 	ldr	w26, [x4, #4]
	sptIndex const stride = mats[0]->stride;
    4830:	b9400c58 	ldr	w24, [x2, #12]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4834:	b9400082 	ldr	w2, [x4]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4838:	a9418ee4 	ldp	x4, x3, [x23, #24]
	sptValue * const restrict mvals = mats[nmodes]->values;
    483c:	f9400819 	ldr	x25, [x0, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4840:	1b027f02 	mul	w2, w24, w2
	sptIndex const * const mode_ind = X->inds[mode].data;
    4844:	9ba50d23 	umaddl	x3, w9, w5, x3
	sptNnzIndex const nnz = X->nnz;
    4848:	f9003fe4 	str	x4, [sp, #120]
	sptValue const * const vals = X->values.data;
    484c:	f9401efc 	ldr	x28, [x23, #56]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4850:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const mode_ind = X->inds[mode].data;
    4854:	f940087b 	ldr	x27, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4858:	aa1903e0 	mov	x0, x25
    485c:	fd0033e8 	str	d8, [sp, #96]
    4860:	97fff29c 	bl	12d0 <memset@plt>
	sptNewTimer(&timer, 0);
    4864:	52800001 	mov	w1, #0x0                   	// #0
    4868:	910203e0 	add	x0, sp, #0x80
    486c:	94000039 	bl	4950 <sptNewTimer>
	sptStartTimer(timer);
    4870:	f94043e0 	ldr	x0, [sp, #128]
    4874:	94000057 	bl	49d0 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    4878:	f9403fe4 	ldr	x4, [sp, #120]
    487c:	90000000 	adrp	x0, 4000 <sptMTTKRP+0x350>
    4880:	2a1403e2 	mov	w2, w20
    4884:	52800003 	mov	w3, #0x0                   	// #0
    4888:	910223e1 	add	x1, sp, #0x88
    488c:	a908cff7 	stp	x23, x19, [sp, #136]
    4890:	91058000 	add	x0, x0, #0x160
    4894:	a90aeffc 	stp	x28, x27, [sp, #168]
    4898:	a90993f5 	stp	x21, x4, [sp, #152]
    489c:	f9005ff9 	str	x25, [sp, #184]
    48a0:	291863f6 	stp	w22, w24, [sp, #192]
    48a4:	b900cbfa 	str	w26, [sp, #200]
    48a8:	97fff2e6 	bl	1440 <GOMP_parallel@plt>
	sptStopTimer(timer);
    48ac:	f94043e0 	ldr	x0, [sp, #128]
    48b0:	94000078 	bl	4a90 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    48b4:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    48b8:	9129c021 	add	x1, x1, #0xa70
    48bc:	f94043e0 	ldr	x0, [sp, #128]
    48c0:	940000b4 	bl	4b90 <sptPrintElapsedTime>
    48c4:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    48c8:	f94043e0 	ldr	x0, [sp, #128]
    48cc:	940000f1 	bl	4c90 <sptFreeTimer>
	printf("[Total time]: %lf\n", total_time);
    48d0:	1e604100 	fmov	d0, d8
    48d4:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    48d8:	9128c000 	add	x0, x0, #0xa30
    48dc:	97fff2d1 	bl	1420 <printf@plt>
	printf("\n");
    48e0:	52800140 	mov	w0, #0xa                   	// #10
    48e4:	97fff2db 	bl	1450 <putchar@plt>
	return 0;
    48e8:	52800000 	mov	w0, #0x0                   	// #0
    48ec:	a94153f3 	ldp	x19, x20, [sp, #16]
}
    48f0:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    48f4:	a94363f7 	ldp	x23, x24, [sp, #48]
    48f8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    48fc:	a94573fb 	ldp	x27, x28, [sp, #80]
    4900:	fd4033e8 	ldr	d8, [sp, #96]
}
    4904:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4908:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    490c:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    4910:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4914:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4918:	91284084 	add	x4, x4, #0xa10
    491c:	91292042 	add	x2, x2, #0xa48
    4920:	9129c000 	add	x0, x0, #0xa70
    4924:	52800883 	mov	w3, #0x44                  	// #68
    4928:	17ffffb3 	b	47f4 <sptOmpMTTKRP+0x94>
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    492c:	97ffff09 	bl	4550 <sptOmpMTTKRP_3D>
    4930:	34fff6c0 	cbz	w0, 4808 <sptOmpMTTKRP+0xa8>
    4934:	12800000 	mov	w0, #0xffffffff            	// #-1
    4938:	a90153f3 	stp	x19, x20, [sp, #16]
    493c:	a90363f7 	stp	x23, x24, [sp, #48]
    4940:	a9046bf9 	stp	x25, x26, [sp, #64]
    4944:	a90573fb 	stp	x27, x28, [sp, #80]
    4948:	fd0033e8 	str	d8, [sp, #96]
    494c:	97fff225 	bl	11e0 <exit@plt>

0000000000004950 <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    4950:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4954:	910003fd 	mov	x29, sp
    4958:	a90153f3 	stp	x19, x20, [sp, #16]
    495c:	aa0003f4 	mov	x20, x0
    4960:	2a0103f3 	mov	w19, w1
	*timer = (sptTimer) malloc(sizeof **timer);
    4964:	d2800500 	mov	x0, #0x28                  	// #40
    4968:	97fff246 	bl	1280 <malloc@plt>
	(*timer)->use_cuda = use_cuda;
    496c:	b9000013 	str	w19, [x0]
	*timer = (sptTimer) malloc(sizeof **timer);
    4970:	f9000280 	str	x0, [x20]
	if(use_cuda) {
    4974:	350000b3 	cbnz	w19, 4988 <sptNewTimer+0x38>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	return 0;
}
    4978:	2a1303e0 	mov	w0, w19
    497c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4980:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4984:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4988:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    498c:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4990:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4994:	912a2084 	add	x4, x4, #0xa88
    4998:	912ac042 	add	x2, x2, #0xab0
    499c:	912b6000 	add	x0, x0, #0xad8
    49a0:	528004e3 	mov	w3, #0x27                  	// #39
    49a4:	52800061 	mov	w1, #0x3                   	// #3
    49a8:	72a00041 	movk	w1, #0x2, lsl #16
    49ac:	52800073 	mov	w19, #0x3                   	// #3
    49b0:	72a00053 	movk	w19, #0x2, lsl #16
    49b4:	94000137 	bl	4e90 <spt_ComplainError>
}
    49b8:	2a1303e0 	mov	w0, w19
    49bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    49c0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    49c4:	d65f03c0 	ret
    49c8:	d503201f 	nop
    49cc:	d503201f 	nop

00000000000049d0 <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    49d0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    49d4:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    49d8:	b9400001 	ldr	w1, [x0]
    49dc:	350000e1 	cbnz	w1, 49f8 <sptStartTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    49e0:	91002001 	add	x1, x0, #0x8
    49e4:	52800020 	mov	w0, #0x1                   	// #1
    49e8:	97fff20a 	bl	1210 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    49ec:	35000220 	cbnz	w0, 4a30 <sptStartTimer+0x60>
	}
	return 0;
}
    49f0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    49f4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    49f8:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    49fc:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4a00:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4a04:	912a2084 	add	x4, x4, #0xa88
    4a08:	912ac042 	add	x2, x2, #0xab0
    4a0c:	912b6000 	add	x0, x0, #0xad8
    4a10:	528005e3 	mov	w3, #0x2f                  	// #47
    4a14:	52800061 	mov	w1, #0x3                   	// #3
    4a18:	72a00041 	movk	w1, #0x2, lsl #16
    4a1c:	9400011d 	bl	4e90 <spt_ComplainError>
}
    4a20:	a8c27bfd 	ldp	x29, x30, [sp], #32
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4a24:	52800060 	mov	w0, #0x3                   	// #3
    4a28:	72a00040 	movk	w0, #0x2, lsl #16
}
    4a2c:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    4a30:	a90153f3 	stp	x19, x20, [sp, #16]
    4a34:	97fff27f 	bl	1430 <__errno_location@plt>
    4a38:	aa0003f3 	mov	x19, x0
	return 0;
    4a3c:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    4a40:	b9400274 	ldr	w20, [x19]
    4a44:	3140429f 	cmn	w20, #0x10, lsl #12
    4a48:	54000061 	b.ne	4a54 <sptStartTimer+0x84>  // b.any
    4a4c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4a50:	17ffffe8 	b	49f0 <sptStartTimer+0x20>
    4a54:	2a1403e0 	mov	w0, w20
    4a58:	97fff236 	bl	1330 <strerror@plt>
    4a5c:	b0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    4a60:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4a64:	aa0003e4 	mov	x4, x0
    4a68:	11404281 	add	w1, w20, #0x10, lsl #12
    4a6c:	912b6060 	add	x0, x3, #0xad8
    4a70:	912ac042 	add	x2, x2, #0xab0
    4a74:	52800643 	mov	w3, #0x32                  	// #50
    4a78:	94000106 	bl	4e90 <spt_ComplainError>
    4a7c:	b9400260 	ldr	w0, [x19]
    4a80:	a94153f3 	ldp	x19, x20, [sp, #16]
    4a84:	11404000 	add	w0, w0, #0x10, lsl #12
    4a88:	17ffffda 	b	49f0 <sptStartTimer+0x20>
    4a8c:	d503201f 	nop

0000000000004a90 <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    4a90:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4a94:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    4a98:	b9400001 	ldr	w1, [x0]
    4a9c:	350000e1 	cbnz	w1, 4ab8 <sptStopTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    4aa0:	91006001 	add	x1, x0, #0x18
    4aa4:	52800020 	mov	w0, #0x1                   	// #1
    4aa8:	97fff1da 	bl	1210 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    4aac:	35000220 	cbnz	w0, 4af0 <sptStopTimer+0x60>
	}
	return 0;
}
    4ab0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4ab4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4ab8:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    4abc:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4ac0:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4ac4:	912a2084 	add	x4, x4, #0xa88
    4ac8:	912ac042 	add	x2, x2, #0xab0
    4acc:	912b6000 	add	x0, x0, #0xad8
    4ad0:	52800743 	mov	w3, #0x3a                  	// #58
    4ad4:	52800061 	mov	w1, #0x3                   	// #3
    4ad8:	72a00041 	movk	w1, #0x2, lsl #16
    4adc:	940000ed 	bl	4e90 <spt_ComplainError>
}
    4ae0:	a8c27bfd 	ldp	x29, x30, [sp], #32
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4ae4:	52800060 	mov	w0, #0x3                   	// #3
    4ae8:	72a00040 	movk	w0, #0x2, lsl #16
}
    4aec:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    4af0:	a90153f3 	stp	x19, x20, [sp, #16]
    4af4:	97fff24f 	bl	1430 <__errno_location@plt>
    4af8:	aa0003f3 	mov	x19, x0
	return 0;
    4afc:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    4b00:	b9400274 	ldr	w20, [x19]
    4b04:	3140429f 	cmn	w20, #0x10, lsl #12
    4b08:	54000061 	b.ne	4b14 <sptStopTimer+0x84>  // b.any
    4b0c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4b10:	17ffffe8 	b	4ab0 <sptStopTimer+0x20>
    4b14:	2a1403e0 	mov	w0, w20
    4b18:	97fff206 	bl	1330 <strerror@plt>
    4b1c:	b0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    4b20:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4b24:	aa0003e4 	mov	x4, x0
    4b28:	11404281 	add	w1, w20, #0x10, lsl #12
    4b2c:	912b6060 	add	x0, x3, #0xad8
    4b30:	912ac042 	add	x2, x2, #0xab0
    4b34:	528007a3 	mov	w3, #0x3d                  	// #61
    4b38:	940000d6 	bl	4e90 <spt_ComplainError>
    4b3c:	b9400260 	ldr	w0, [x19]
    4b40:	a94153f3 	ldp	x19, x20, [sp, #16]
    4b44:	11404000 	add	w0, w0, #0x10, lsl #12
    4b48:	17ffffda 	b	4ab0 <sptStopTimer+0x20>
    4b4c:	d503201f 	nop

0000000000004b50 <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
	if(timer->use_cuda) {
    4b50:	b9400001 	ldr	w1, [x0]
    4b54:	35000161 	cbnz	w1, 4b80 <sptElapsedTime+0x30>
		return NAN;
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4b58:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4b5c:	a9408c04 	ldp	x4, x3, [x0, #8]
    4b60:	fd431422 	ldr	d2, [x1, #1576]
    4b64:	a9418801 	ldp	x1, x2, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4b68:	cb040020 	sub	x0, x1, x4
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4b6c:	cb030041 	sub	x1, x2, x3
    4b70:	9e620000 	scvtf	d0, x0
    4b74:	9e620021 	scvtf	d1, x1
    4b78:	1f420020 	fmadd	d0, d1, d2, d0
	}
}
    4b7c:	d65f03c0 	ret
		return NAN;
    4b80:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4b84:	9e670000 	fmov	d0, x0
}
    4b88:	d65f03c0 	ret
    4b8c:	d503201f 	nop

0000000000004b90 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4b90:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4b94:	910003fd 	mov	x29, sp
    4b98:	aa0103e2 	mov	x2, x1
	if(timer->use_cuda) {
    4b9c:	b9400001 	ldr	w1, [x0]
double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4ba0:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4ba4:	350002a1 	cbnz	w1, 4bf8 <sptPrintElapsedTime+0x68>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4ba8:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4bac:	a9409005 	ldp	x5, x4, [x0, #8]
    4bb0:	fd431421 	ldr	d1, [x1, #1576]
    4bb4:	a9418403 	ldp	x3, x1, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4bb8:	cb050060 	sub	x0, x3, x5
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4bbc:	cb040021 	sub	x1, x1, x4
    4bc0:	9e620008 	scvtf	d8, x0
    4bc4:	9e620020 	scvtf	d0, x1
    4bc8:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer);
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4bcc:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    4bd0:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4bd4:	1e604100 	fmov	d0, d8
    4bd8:	f947e000 	ldr	x0, [x0, #4032]
    4bdc:	912ba021 	add	x1, x1, #0xae8
    4be0:	f9400000 	ldr	x0, [x0]
    4be4:	97fff21f 	bl	1460 <fprintf@plt>
	return elapsed_time;
}
    4be8:	1e604100 	fmov	d0, d8
    4bec:	fd400be8 	ldr	d8, [sp, #16]
    4bf0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4bf4:	d65f03c0 	ret
		return NAN;
    4bf8:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4bfc:	9e670008 	fmov	d8, x0
    4c00:	17fffff3 	b	4bcc <sptPrintElapsedTime+0x3c>
    4c04:	d503201f 	nop
    4c08:	d503201f 	nop
    4c0c:	d503201f 	nop

0000000000004c10 <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4c10:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4c14:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    4c18:	b9400003 	ldr	w3, [x0]
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4c1c:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4c20:	350002e3 	cbnz	w3, 4c7c <sptPrintAverageElapsedTime+0x6c>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4c24:	b0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    4c28:	a9409406 	ldp	x6, x5, [x0, #8]
    4c2c:	fd431461 	ldr	d1, [x3, #1576]
    4c30:	a9419003 	ldp	x3, x4, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4c34:	cb060060 	sub	x0, x3, x6
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4c38:	cb050083 	sub	x3, x4, x5
    4c3c:	9e620008 	scvtf	d8, x0
    4c40:	9e620060 	scvtf	d0, x3
    4c44:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer) / niters;
    4c48:	1e620020 	scvtf	d0, w1
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4c4c:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    4c50:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4c54:	f947e000 	ldr	x0, [x0, #4032]
    4c58:	912ba021 	add	x1, x1, #0xae8
	double elapsed_time = sptElapsedTime(timer) / niters;
    4c5c:	1e601908 	fdiv	d8, d8, d0
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4c60:	f9400000 	ldr	x0, [x0]
    4c64:	1e604100 	fmov	d0, d8
    4c68:	97fff1fe 	bl	1460 <fprintf@plt>
	return elapsed_time;
}
    4c6c:	1e604100 	fmov	d0, d8
    4c70:	fd400be8 	ldr	d8, [sp, #16]
    4c74:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4c78:	d65f03c0 	ret
		return NAN;
    4c7c:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4c80:	9e670008 	fmov	d8, x0
    4c84:	17fffff1 	b	4c48 <sptPrintAverageElapsedTime+0x38>
    4c88:	d503201f 	nop
    4c8c:	d503201f 	nop

0000000000004c90 <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    4c90:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4c94:	910003fd 	mov	x29, sp
    4c98:	f9000bf3 	str	x19, [sp, #16]
	if(timer->use_cuda) {
    4c9c:	b9400013 	ldr	w19, [x0]
    4ca0:	350000d3 	cbnz	w19, 4cb8 <sptFreeTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	free(timer);
    4ca4:	97fff1c7 	bl	13c0 <free@plt>
	return 0;
}
    4ca8:	2a1303e0 	mov	w0, w19
    4cac:	f9400bf3 	ldr	x19, [sp, #16]
    4cb0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4cb4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4cb8:	b0000004 	adrp	x4, 5000 <sptBytesString+0x80>
    4cbc:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4cc0:	b0000000 	adrp	x0, 5000 <sptBytesString+0x80>
    4cc4:	912a2084 	add	x4, x4, #0xa88
    4cc8:	912ac042 	add	x2, x2, #0xab0
    4ccc:	912b6000 	add	x0, x0, #0xad8
    4cd0:	52800b63 	mov	w3, #0x5b                  	// #91
    4cd4:	52800061 	mov	w1, #0x3                   	// #3
    4cd8:	72a00041 	movk	w1, #0x2, lsl #16
    4cdc:	52800073 	mov	w19, #0x3                   	// #3
    4ce0:	72a00053 	movk	w19, #0x2, lsl #16
    4ce4:	9400006b 	bl	4e90 <spt_ComplainError>
}
    4ce8:	2a1303e0 	mov	w0, w19
    4cec:	f9400bf3 	ldr	x19, [sp, #16]
    4cf0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4cf4:	d65f03c0 	ret
    4cf8:	d503201f 	nop
    4cfc:	d503201f 	nop

0000000000004d00 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4d00:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
    4d04:	910003fd 	mov	x29, sp
    4d08:	a9025bf5 	stp	x21, x22, [sp, #32]
    4d0c:	aa0003f6 	mov	x22, x0
    4d10:	aa0103f5 	mov	x21, x1
	int iores;
	sptIndex nrows = mtx->nrows;
	sptIndex ncols = mtx->ncols;
	sptIndex stride = mtx->stride;
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4d14:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4d18:	912be021 	add	x1, x1, #0xaf8
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4d1c:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex nrows = mtx->nrows;
    4d20:	b940001a 	ldr	w26, [x0]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4d24:	aa1503e0 	mov	x0, x21
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4d28:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex ncols = mtx->ncols;
    4d2c:	b94006d9 	ldr	w25, [x22, #4]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4d30:	2a1a03e2 	mov	w2, w26
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4d34:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex stride = mtx->stride;
    4d38:	b9400edb 	ldr	w27, [x22, #12]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4d3c:	2a1903e3 	mov	w3, w25
    4d40:	97fff1c8 	bl	1460 <fprintf@plt>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4d44:	37f80800 	tbnz	w0, #31, 4e44 <sptDumpMatrix+0x144>
	for(sptIndex i=0; i < nrows; ++i) {
    4d48:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex ncols = mtx->ncols;
    4d4c:	2a1903f4 	mov	w20, w25
    4d50:	b0000017 	adrp	x23, 5000 <sptBytesString+0x80>
		for(sptIndex j=0; j < ncols; ++j) {
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4d54:	912d02f7 	add	x23, x23, #0xb40
	for(sptIndex i=0; i < nrows; ++i) {
    4d58:	52800018 	mov	w24, #0x0                   	// #0
    4d5c:	340005fa 	cbz	w26, 4e18 <sptDumpMatrix+0x118>
		for(sptIndex j=0; j < ncols; ++j) {
    4d60:	4b190293 	sub	w19, w20, w25
    4d64:	35000099 	cbnz	w25, 4d74 <sptDumpMatrix+0x74>
    4d68:	14000025 	b	4dfc <sptDumpMatrix+0xfc>
    4d6c:	6b13029f 	cmp	w20, w19
    4d70:	54000460 	b.eq	4dfc <sptDumpMatrix+0xfc>  // b.none
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4d74:	f9400ac2 	ldr	x2, [x22, #16]
    4d78:	aa1703e1 	mov	x1, x23
    4d7c:	aa1503e0 	mov	x0, x21
    4d80:	bc735840 	ldr	s0, [x2, w19, uxtw #2]
		for(sptIndex j=0; j < ncols; ++j) {
    4d84:	11000673 	add	w19, w19, #0x1
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4d88:	1e22c000 	fcvt	d0, s0
    4d8c:	97fff1b5 	bl	1460 <fprintf@plt>
			spt_CheckOSError(iores < 0, "Mtx Dump");
    4d90:	36fffee0 	tbz	w0, #31, 4d6c <sptDumpMatrix+0x6c>
    4d94:	97fff1a7 	bl	1430 <__errno_location@plt>
    4d98:	b9400001 	ldr	w1, [x0]
    4d9c:	aa0003fc 	mov	x28, x0
    4da0:	3140403f 	cmn	w1, #0x10, lsl #12
    4da4:	54fffe40 	b.eq	4d6c <sptDumpMatrix+0x6c>  // b.none
    4da8:	2a0103e0 	mov	w0, w1
    4dac:	b9006fe1 	str	w1, [sp, #108]
    4db0:	97fff160 	bl	1330 <strerror@plt>
    4db4:	b9406fe1 	ldr	w1, [sp, #108]
    4db8:	b0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    4dbc:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4dc0:	aa0003e4 	mov	x4, x0
    4dc4:	912c2042 	add	x2, x2, #0xb08
    4dc8:	912cc060 	add	x0, x3, #0xb30
    4dcc:	528005c3 	mov	w3, #0x2e                  	// #46
    4dd0:	11404021 	add	w1, w1, #0x10, lsl #12
    4dd4:	9400002f 	bl	4e90 <spt_ComplainError>
    4dd8:	b9400380 	ldr	w0, [x28]
    4ddc:	a94363f7 	ldp	x23, x24, [sp, #48]
    4de0:	11404000 	add	w0, w0, #0x10, lsl #12
		}
		iores = fprintf(fp, "\n");
	}
	iores = fprintf(fp, "\n");
	return 0;
}
    4de4:	a94153f3 	ldp	x19, x20, [sp, #16]
    4de8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4dec:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4df0:	a94573fb 	ldp	x27, x28, [sp, #80]
    4df4:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4df8:	d65f03c0 	ret
		iores = fprintf(fp, "\n");
    4dfc:	aa1503e1 	mov	x1, x21
    4e00:	52800140 	mov	w0, #0xa                   	// #10
	for(sptIndex i=0; i < nrows; ++i) {
    4e04:	11000718 	add	w24, w24, #0x1
    4e08:	0b1b0294 	add	w20, w20, w27
		iores = fprintf(fp, "\n");
    4e0c:	97fff0fd 	bl	1200 <fputc@plt>
	for(sptIndex i=0; i < nrows; ++i) {
    4e10:	6b18035f 	cmp	w26, w24
    4e14:	54fffa61 	b.ne	4d60 <sptDumpMatrix+0x60>  // b.any
	iores = fprintf(fp, "\n");
    4e18:	aa1503e1 	mov	x1, x21
    4e1c:	52800140 	mov	w0, #0xa                   	// #10
    4e20:	97fff0f8 	bl	1200 <fputc@plt>
	return 0;
    4e24:	52800000 	mov	w0, #0x0                   	// #0
}
    4e28:	a94153f3 	ldp	x19, x20, [sp, #16]
    4e2c:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    4e30:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    4e34:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4e38:	a94573fb 	ldp	x27, x28, [sp, #80]
    4e3c:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4e40:	d65f03c0 	ret
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4e44:	97fff17b 	bl	1430 <__errno_location@plt>
    4e48:	b9400014 	ldr	w20, [x0]
    4e4c:	aa0003f3 	mov	x19, x0
    4e50:	3140429f 	cmn	w20, #0x10, lsl #12
    4e54:	54fff7a0 	b.eq	4d48 <sptDumpMatrix+0x48>  // b.none
    4e58:	2a1403e0 	mov	w0, w20
    4e5c:	97fff135 	bl	1330 <strerror@plt>
    4e60:	b0000003 	adrp	x3, 5000 <sptBytesString+0x80>
    4e64:	b0000002 	adrp	x2, 5000 <sptBytesString+0x80>
    4e68:	aa0003e4 	mov	x4, x0
    4e6c:	11404281 	add	w1, w20, #0x10, lsl #12
    4e70:	912cc060 	add	x0, x3, #0xb30
    4e74:	912c2042 	add	x2, x2, #0xb08
    4e78:	52800543 	mov	w3, #0x2a                  	// #42
    4e7c:	94000005 	bl	4e90 <spt_ComplainError>
    4e80:	b9400260 	ldr	w0, [x19]
    4e84:	11404000 	add	w0, w0, #0x10, lsl #12
    4e88:	17ffffd7 	b	4de4 <sptDumpMatrix+0xe4>
    4e8c:	d503201f 	nop

0000000000004e90 <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4e90:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4e94:	910003fd 	mov	x29, sp
    4e98:	a90153f3 	stp	x19, x20, [sp, #16]
	g_last_error.errcode = errcode;
    4e9c:	d53bd053 	mrs	x19, tpidr_el0
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4ea0:	aa0003f4 	mov	x20, x0
	g_last_error.errcode = errcode;
    4ea4:	91400273 	add	x19, x19, #0x0, lsl #12
    4ea8:	91004273 	add	x19, x19, #0x10
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4eac:	a9025bf5 	stp	x21, x22, [sp, #32]
    4eb0:	2a0103f5 	mov	w21, w1
    4eb4:	aa0203f6 	mov	x22, x2
    4eb8:	a90363f7 	stp	x23, x24, [sp, #48]
    4ebc:	2a0303f7 	mov	w23, w3
	g_last_error.module = module;
    4ec0:	f9000260 	str	x0, [x19]
	g_last_error.errcode = errcode;
    4ec4:	b9000a61 	str	w1, [x19, #8]
	g_last_error.file = file;
    4ec8:	f9000a62 	str	x2, [x19, #16]
	g_last_error.line = line;
    4ecc:	b9001a63 	str	w3, [x19, #24]
	if(reason) {
		free(g_last_error.reason);
    4ed0:	f9401266 	ldr	x6, [x19, #32]
	if(reason) {
    4ed4:	b4000324 	cbz	x4, 4f38 <spt_ComplainError+0xa8>
		free(g_last_error.reason);
    4ed8:	aa0603e0 	mov	x0, x6
    4edc:	aa0403f8 	mov	x24, x4
    4ee0:	97fff138 	bl	13c0 <free@plt>
		g_last_error.reason = strdup(reason);
    4ee4:	aa1803e0 	mov	x0, x24
    4ee8:	97fff10a 	bl	1310 <strdup@plt>
    4eec:	f9001260 	str	x0, [x19, #32]
    4ef0:	aa0003e6 	mov	x6, x0
		if(!g_last_error.reason) {
    4ef4:	b4000440 	cbz	x0, 4f7c <spt_ComplainError+0xec>
			abort();
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4ef8:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf684>
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4efc:	394000c1 	ldrb	w1, [x6]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4f00:	f947d800 	ldr	x0, [x0, #4016]
    4f04:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4f08:	35000241 	cbnz	w1, 4f50 <spt_ComplainError+0xc0>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4f0c:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4f10:	2a1703e5 	mov	w5, w23
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    4f14:	a94363f7 	ldp	x23, x24, [sp, #48]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4f18:	aa1603e4 	mov	x4, x22
    4f1c:	2a1503e3 	mov	w3, w21
    4f20:	aa1403e2 	mov	x2, x20
    4f24:	912da021 	add	x1, x1, #0xb68
}
    4f28:	a94153f3 	ldp	x19, x20, [sp, #16]
    4f2c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4f30:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4f34:	17fff14b 	b	1460 <fprintf@plt>
    4f38:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    4f3c:	f947d800 	ldr	x0, [x0, #4016]
    4f40:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4f44:	b4fffe46 	cbz	x6, 4f0c <spt_ComplainError+0x7c>
    4f48:	394000c1 	ldrb	w1, [x6]
    4f4c:	34fffe01 	cbz	w1, 4f0c <spt_ComplainError+0x7c>
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4f50:	b0000001 	adrp	x1, 5000 <sptBytesString+0x80>
    4f54:	2a1703e5 	mov	w5, w23
}
    4f58:	a94363f7 	ldp	x23, x24, [sp, #48]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4f5c:	aa1603e4 	mov	x4, x22
    4f60:	2a1503e3 	mov	w3, w21
    4f64:	aa1403e2 	mov	x2, x20
    4f68:	912d2021 	add	x1, x1, #0xb48
}
    4f6c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4f70:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4f74:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4f78:	17fff13a 	b	1460 <fprintf@plt>
			abort();
    4f7c:	97fff0fd 	bl	1370 <abort@plt>

0000000000004f80 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    4f80:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    4f84:	910003fd 	mov	x29, sp
	double size = (double)bytes;
	int suff = 0;
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
	while(size > 1024 && suff < 5) {
    4f88:	d2e81201 	mov	x1, #0x4090000000000000    	// #4652218415073722368
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4f8c:	9100a3e2 	add	x2, sp, #0x28
{
    4f90:	fd000be8 	str	d8, [sp, #16]
    4f94:	9e670008 	fmov	d8, x0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4f98:	f0000080 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    4f9c:	910a0000 	add	x0, x0, #0x280
	while(size > 1024 && suff < 5) {
    4fa0:	9e670021 	fmov	d1, x1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4fa4:	a9401c06 	ldp	x6, x7, [x0]
    4fa8:	a9411404 	ldp	x4, x5, [x0, #16]
    4fac:	a9029fe6 	stp	x6, x7, [sp, #40]
	double size = (double)bytes;
    4fb0:	7e61d900 	ucvtf	d0, d8
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4fb4:	f9401000 	ldr	x0, [x0, #32]
    4fb8:	a9011444 	stp	x4, x5, [x2, #16]
	while(size > 1024 && suff < 5) {
    4fbc:	1e612010 	fcmpe	d0, d1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4fc0:	f9001040 	str	x0, [x2, #32]
	while(size > 1024 && suff < 5) {
    4fc4:	54000669 	b.ls	5090 <sptBytesString+0x110>  // b.plast
    4fc8:	d2e82601 	mov	x1, #0x4130000000000000    	// #4697254411347427328
		size /= 1024.;
    4fcc:	d2e7ea00 	mov	x0, #0x3f50000000000000    	// #4562146422526312448
	while(size > 1024 && suff < 5) {
    4fd0:	9e670022 	fmov	d2, x1
		size /= 1024.;
    4fd4:	9e670001 	fmov	d1, x0
	while(size > 1024 && suff < 5) {
    4fd8:	1e622010 	fcmpe	d0, d2
		size /= 1024.;
    4fdc:	1e610802 	fmul	d2, d0, d1
	while(size > 1024 && suff < 5) {
    4fe0:	540005c9 	b.ls	5098 <sptBytesString+0x118>  // b.plast
    4fe4:	d2e83a00 	mov	x0, #0x41d0000000000000    	// #4742290407621132288
		size /= 1024.;
    4fe8:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4fec:	9e670003 	fmov	d3, x0
    4ff0:	1e632010 	fcmpe	d0, d3
    4ff4:	54000589 	b.ls	50a4 <sptBytesString+0x124>  // b.plast
    4ff8:	d2e84e00 	mov	x0, #0x4270000000000000    	// #4787326403894837248
		size /= 1024.;
    4ffc:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    5000:	9e670003 	fmov	d3, x0
    5004:	1e632010 	fcmpe	d0, d3
    5008:	540003e9 	b.ls	5084 <sptBytesString+0x104>  // b.plast
    500c:	d2e86200 	mov	x0, #0x4310000000000000    	// #4832362400168542208
		size /= 1024.;
    5010:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    5014:	9e670003 	fmov	d3, x0
    5018:	1e632010 	fcmpe	d0, d3
    501c:	540004a9 	b.ls	50b0 <sptBytesString+0x130>  // b.plast
		size /= 1024.;
    5020:	1e610840 	fmul	d0, d2, d1
		++suff;
    5024:	528000a3 	mov	w3, #0x5                   	// #5
	}
	char * ret = NULL;
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    5028:	90000001 	adrp	x1, 5000 <sptBytesString+0x80>
    502c:	910083e0 	add	x0, sp, #0x20
    5030:	f863d842 	ldr	x2, [x2, w3, sxtw #3]
    5034:	912ec021 	add	x1, x1, #0xbb0
	char * ret = NULL;
    5038:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    503c:	97fff079 	bl	1220 <asprintf@plt>
    5040:	3100041f 	cmn	w0, #0x1
    5044:	540000a0 	b.eq	5058 <sptBytesString+0xd8>  // b.none
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
		ret = NULL;
	}
	return ret;
}
    5048:	fd400be8 	ldr	d8, [sp, #16]
	return ret;
    504c:	f94013e0 	ldr	x0, [sp, #32]
}
    5050:	a8c57bfd 	ldp	x29, x30, [sp], #80
    5054:	d65f03c0 	ret
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    5058:	b0000080 	adrp	x0, 16000 <__FRAME_END__+0xf684>
    505c:	90000001 	adrp	x1, 5000 <sptBytesString+0x80>
    5060:	f947d800 	ldr	x0, [x0, #4016]
    5064:	912ee021 	add	x1, x1, #0xbb8
    5068:	9e660102 	fmov	x2, d8
    506c:	f9400000 	ldr	x0, [x0]
    5070:	97fff0fc 	bl	1460 <fprintf@plt>
    5074:	d2800000 	mov	x0, #0x0                   	// #0
}
    5078:	fd400be8 	ldr	d8, [sp, #16]
    507c:	a8c57bfd 	ldp	x29, x30, [sp], #80
    5080:	d65f03c0 	ret
		size /= 1024.;
    5084:	1e604040 	fmov	d0, d2
		++suff;
    5088:	52800063 	mov	w3, #0x3                   	// #3
    508c:	17ffffe7 	b	5028 <sptBytesString+0xa8>
	int suff = 0;
    5090:	52800003 	mov	w3, #0x0                   	// #0
    5094:	17ffffe5 	b	5028 <sptBytesString+0xa8>
		size /= 1024.;
    5098:	1e604040 	fmov	d0, d2
		++suff;
    509c:	52800023 	mov	w3, #0x1                   	// #1
    50a0:	17ffffe2 	b	5028 <sptBytesString+0xa8>
		size /= 1024.;
    50a4:	1e604040 	fmov	d0, d2
		++suff;
    50a8:	52800043 	mov	w3, #0x2                   	// #2
    50ac:	17ffffdf 	b	5028 <sptBytesString+0xa8>
		size /= 1024.;
    50b0:	1e604040 	fmov	d0, d2
		++suff;
    50b4:	52800083 	mov	w3, #0x4                   	// #4
    50b8:	17ffffdc 	b	5028 <sptBytesString+0xa8>
    50bc:	d503201f 	nop

00000000000050c0 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    50c0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    50c4:	910003fd 	mov	x29, sp
    50c8:	fd000be8 	str	d8, [sp, #16]
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    50cc:	97fff08d 	bl	1300 <rand@plt>
    50d0:	52a61801 	mov	w1, #0x30c00000            	// #817889280
    50d4:	1e220008 	scvtf	s8, w0
    50d8:	1e270020 	fmov	s0, w1
    50dc:	1e200908 	fmul	s8, s8, s0
	if(rand() % 2 == 0) {
    50e0:	97fff088 	bl	1300 <rand@plt>
		v *= -1;
    50e4:	1e214100 	fneg	s0, s8
    50e8:	f240001f 	tst	x0, #0x1
	}
	return v;
}
    50ec:	1e280c00 	fcsel	s0, s0, s8, eq  // eq = none
    50f0:	fd400be8 	ldr	d8, [sp, #16]
    50f4:	a8c27bfd 	ldp	x29, x30, [sp], #32
    50f8:	d65f03c0 	ret
    50fc:	d503201f 	nop

0000000000005100 <__aarch64_cas4_acq_rel>:
    5100:	d503245f 	bti	c
    5104:	d0000090 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    5108:	394aa610 	ldrb	w16, [x16, #681]
    510c:	34000070 	cbz	w16, 5118 <__aarch64_cas4_acq_rel+0x18>
    5110:	88e0fc41 	casal	w0, w1, [x2]
    5114:	d65f03c0 	ret
    5118:	2a0003f0 	mov	w16, w0
    511c:	885ffc40 	ldaxr	w0, [x2]
    5120:	6b10001f 	cmp	w0, w16
    5124:	54000061 	b.ne	5130 <__aarch64_cas4_acq_rel+0x30>  // b.any
    5128:	8811fc41 	stlxr	w17, w1, [x2]
    512c:	35ffff91 	cbnz	w17, 511c <__aarch64_cas4_acq_rel+0x1c>
    5130:	d65f03c0 	ret
    5134:	d503201f 	nop
    5138:	d503201f 	nop
    513c:	d503201f 	nop

0000000000005140 <__libc_csu_init>:
    5140:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    5144:	910003fd 	mov	x29, sp
    5148:	a90153f3 	stp	x19, x20, [sp, #16]
    514c:	b0000094 	adrp	x20, 16000 <__FRAME_END__+0xf684>
    5150:	9134e294 	add	x20, x20, #0xd38
    5154:	a9025bf5 	stp	x21, x22, [sp, #32]
    5158:	b0000095 	adrp	x21, 16000 <__FRAME_END__+0xf684>
    515c:	913482b5 	add	x21, x21, #0xd20
    5160:	cb150294 	sub	x20, x20, x21
    5164:	2a0003f6 	mov	w22, w0
    5168:	a90363f7 	stp	x23, x24, [sp, #48]
    516c:	aa0103f7 	mov	x23, x1
    5170:	aa0203f8 	mov	x24, x2
    5174:	9343fe94 	asr	x20, x20, #3
    5178:	97fff006 	bl	1190 <_init>
    517c:	b4000174 	cbz	x20, 51a8 <__libc_csu_init+0x68>
    5180:	d2800013 	mov	x19, #0x0                   	// #0
    5184:	d503201f 	nop
    5188:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    518c:	aa1803e2 	mov	x2, x24
    5190:	91000673 	add	x19, x19, #0x1
    5194:	aa1703e1 	mov	x1, x23
    5198:	2a1603e0 	mov	w0, w22
    519c:	d63f0060 	blr	x3
    51a0:	eb13029f 	cmp	x20, x19
    51a4:	54ffff21 	b.ne	5188 <__libc_csu_init+0x48>  // b.any
    51a8:	a94153f3 	ldp	x19, x20, [sp, #16]
    51ac:	a9425bf5 	ldp	x21, x22, [sp, #32]
    51b0:	a94363f7 	ldp	x23, x24, [sp, #48]
    51b4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    51b8:	d65f03c0 	ret
    51bc:	d503201f 	nop

00000000000051c0 <__libc_csu_fini>:
    51c0:	d65f03c0 	ret

Disassembly of section .fini:

00000000000051c4 <_fini>:
    51c4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    51c8:	910003fd 	mov	x29, sp
    51cc:	a8c17bfd 	ldp	x29, x30, [sp], #16
    51d0:	d65f03c0 	ret
