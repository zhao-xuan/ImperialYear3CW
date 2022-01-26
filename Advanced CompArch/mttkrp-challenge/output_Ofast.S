
mttkrp:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000001190 <_init>:
    1190:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1194:	910003fd 	mov	x29, sp
    1198:	94000297 	bl	1bf4 <call_weak_fn>
    119c:	a8c17bfd 	ldp	x29, x30, [sp], #16
    11a0:	d65f03c0 	ret

Disassembly of section .plt:

00000000000011b0 <.plt>:
    11b0:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
    11b4:	b00000b0 	adrp	x16, 16000 <__FRAME_END__+0xf98c>
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
{
    147c:	a9007bfd 	stp	x29, x30, [sp]
    1480:	910003fd 	mov	x29, sp
    1484:	a90153f3 	stp	x19, x20, [sp, #16]
    1488:	2a0003f3 	mov	w19, w0
    148c:	aa0103f4 	mov	x20, x1
	int niters = 5;
	int nthreads = 1;
	printf("niters: %d\n", niters);
    1490:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1494:	528000a1 	mov	w1, #0x5                   	// #5
    1498:	91040000 	add	x0, x0, #0x100
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
    14bc:	540035ad 	b.le	1b70 <main+0x700>
			case 'i':
				strcpy(fname, optarg);
				printf("input file: %s\n", fname); fflush(stdout);
				break;
			case 'o':
				fo = fopen(optarg, "w");
    14c0:	b00000ba 	adrp	x26, 16000 <__FRAME_END__+0xf98c>
    14c4:	d00000b7 	adrp	x23, 17000 <memcpy@GLIBC_2.17>
    14c8:	9000003c 	adrp	x28, 5000 <_IO_stdin_used+0x120>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14cc:	910582f7 	add	x23, x23, #0x160
				fo = fopen(optarg, "w");
    14d0:	f947df5b 	ldr	x27, [x26, #4024]
    14d4:	911fa39c 	add	x28, x28, #0x7e8
    14d8:	9101f3f9 	add	x25, sp, #0x7c
	bool random = true;
    14dc:	52800035 	mov	w21, #0x1                   	// #1
	FILE *fo = NULL;
    14e0:	d2800016 	mov	x22, #0x0                   	// #0
    14e4:	90000038 	adrp	x24, 5000 <_IO_stdin_used+0x120>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14e8:	aa1903e4 	mov	x4, x25
    14ec:	aa1703e3 	mov	x3, x23
    14f0:	91044302 	add	x2, x24, #0x110
    14f4:	aa1403e1 	mov	x1, x20
    14f8:	2a1303e0 	mov	w0, w19
		int option_index = 0;
    14fc:	b9007fff 	str	wzr, [sp, #124]
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
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
    152c:	9101c3e2 	add	x2, sp, #0x70
    1530:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    1534:	91052021 	add	x1, x1, #0x148
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
    1550:	9101d3e2 	add	x2, sp, #0x74
    1554:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    1558:	9105e021 	add	x1, x1, #0x178
    155c:	f9400000 	ldr	x0, [x0]
    1560:	97ffffac 	bl	1410 <__isoc99_sscanf@plt>
				break;
    1564:	17ffffe1 	b	14e8 <main+0x78>
		switch(c) {
    1568:	7101901f 	cmp	w0, #0x64
    156c:	54003021 	b.ne	1b70 <main+0x700>  // b.any
				sscanf(optarg, "%d", &dev_id);
    1570:	f947df40 	ldr	x0, [x26, #4024]
    1574:	9101e3e2 	add	x2, sp, #0x78
    1578:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    157c:	91054021 	add	x1, x1, #0x150
    1580:	f9400000 	ldr	x0, [x0]
    1584:	97ffffa3 	bl	1410 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    1588:	b9407be0 	ldr	w0, [sp, #120]
    158c:	11000800 	add	w0, w0, #0x2
    1590:	7100041f 	cmp	w0, #0x1
    1594:	54fffaa9 	b.ls	14e8 <main+0x78>  // b.plast
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    1598:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xf98c>
    159c:	d2800382 	mov	x2, #0x1c                  	// #28
    15a0:	d2800021 	mov	x1, #0x1                   	// #1
    15a4:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    15a8:	f947d863 	ldr	x3, [x3, #4016]
    15ac:	91056000 	add	x0, x0, #0x158
    15b0:	f9400063 	ldr	x3, [x3]
    15b4:	97ffff87 	bl	13d0 <fwrite@plt>
					exit(1);
    15b8:	52800020 	mov	w0, #0x1                   	// #1
    15bc:	97ffff09 	bl	11e0 <exit@plt>
		switch(c) {
    15c0:	7101d81f 	cmp	w0, #0x76
    15c4:	54002d61 	b.ne	1b70 <main+0x700>  // b.any
			case 'v':
				random = false;
				strcpy(fvname, optarg);
    15c8:	f947df41 	ldr	x1, [x26, #4024]
    15cc:	9112c3e2 	add	x2, sp, #0x4b0
    15d0:	aa0203e0 	mov	x0, x2
				random = false;
    15d4:	52800015 	mov	w21, #0x0                   	// #0
				strcpy(fvname, optarg);
    15d8:	f9400021 	ldr	x1, [x1]
    15dc:	97ffff85 	bl	13f0 <strcpy@plt>
    15e0:	aa0003e2 	mov	x2, x0
				printf("validation input file: %s\n", fvname); fflush(stdout);
    15e4:	aa0203e1 	mov	x1, x2
    15e8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    15ec:	91060000 	add	x0, x0, #0x180
				printf("output file: %s\n", optarg); fflush(stdout);
    15f0:	97ffff8c 	bl	1420 <printf@plt>
    15f4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
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
    1620:	aa0203e1 	mov	x1, x2
    1624:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1628:	91048000 	add	x0, x0, #0x120
    162c:	17fffff1 	b	15f0 <main+0x180>
				fo = fopen(optarg, "w");
    1630:	f9400360 	ldr	x0, [x27]
    1634:	aa1c03e1 	mov	x1, x28
    1638:	97ffff0a 	bl	1260 <fopen@plt>
    163c:	aa0003f6 	mov	x22, x0
				strcpy(foname, optarg);
    1640:	f9400361 	ldr	x1, [x27]
    1644:	912263e0 	add	x0, sp, #0x898
    1648:	f90037e1 	str	x1, [sp, #104]
    164c:	97ffff69 	bl	13f0 <strcpy@plt>
				sptAssert(fo != NULL);
    1650:	f94037e1 	ldr	x1, [sp, #104]
    1654:	b4001cd6 	cbz	x22, 19ec <main+0x57c>
				printf("output file: %s\n", optarg); fflush(stdout);
    1658:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    165c:	9104c000 	add	x0, x0, #0x130
    1660:	17ffffe4 	b	15f0 <main+0x180>
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1664:	b94073e1 	ldr	w1, [sp, #112]
    1668:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    166c:	91068000 	add	x0, x0, #0x1a0
	printf("dev_id: %d\n", dev_id);

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1670:	910223f9 	add	x25, sp, #0x88
	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1674:	97ffff6b 	bl	1420 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    1678:	b9407be1 	ldr	w1, [sp, #120]
    167c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1680:	9106c000 	add	x0, x0, #0x1b0
    1684:	97ffff67 	bl	1420 <printf@plt>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1688:	aa1903e0 	mov	x0, x25
    168c:	910323e2 	add	x2, sp, #0xc8
    1690:	52800021 	mov	w1, #0x1                   	// #1
    1694:	9400042c 	bl	2744 <sptLoadSparseTensor>
    1698:	35001aa0 	cbnz	w0, 19ec <main+0x57c>
	sptSparseTensorStatus(&X, stdout);
    169c:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xf98c>
    16a0:	aa1903e0 	mov	x0, x25
    16a4:	f947e021 	ldr	x1, [x1, #4032]
    16a8:	f9400021 	ldr	x1, [x1]
    16ac:	9400079d 	bl	3520 <sptSparseTensorStatus>

	sptIndex nmodes = X.nmodes;
    16b0:	b9408bf4 	ldr	w20, [sp, #136]
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    16b4:	11000680 	add	w0, w20, #0x1
    16b8:	1100069a 	add	w26, w20, #0x1
    16bc:	d37df000 	lsl	x0, x0, #3
    16c0:	97fffef0 	bl	1280 <malloc@plt>
    16c4:	aa0003f3 	mov	x19, x0
	for(sptIndex m=0; m<nmodes+1; ++m) {
    16c8:	aa0003f7 	mov	x23, x0
    16cc:	8b3a4c18 	add	x24, x0, w26, uxtw #3
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
    16f0:	340002b4 	cbz	w20, 1744 <main+0x2d4>
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    16f4:	f9404fe0 	ldr	x0, [sp, #152]
    16f8:	d2800017 	mov	x23, #0x0                   	// #0
    16fc:	52800018 	mov	w24, #0x0                   	// #0
    1700:	b8777801 	ldr	w1, [x0, x23, lsl #2]
    1704:	d37ef6fa 	lsl	x26, x23, #2
    1708:	b94077e2 	ldr	w2, [sp, #116]
    170c:	f8777a60 	ldr	x0, [x19, x23, lsl #3]
    1710:	9400069c 	bl	3180 <sptNewMatrix>
    1714:	350016c0 	cbnz	w0, 19ec <main+0x57c>
		// sptAssert(sptConstantMatrix(U[m], 1) == 0);
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    1718:	f8777a60 	ldr	x0, [x19, x23, lsl #3]
    171c:	2a1503e1 	mov	w1, w21
    1720:	940006c8 	bl	3240 <sptRandomizeMatrix>
    1724:	35001640 	cbnz	w0, 19ec <main+0x57c>
		if(X.ndims[m] > max_ndims)
    1728:	f9404fe0 	ldr	x0, [sp, #152]
	for(sptIndex m=0; m<nmodes; ++m) {
    172c:	910006f7 	add	x23, x23, #0x1
    1730:	b87a6801 	ldr	w1, [x0, x26]
    1734:	6b01031f 	cmp	w24, w1
    1738:	1a812318 	csel	w24, w24, w1, cs  // cs = hs, nlast
    173c:	6b17029f 	cmp	w20, w23
    1740:	54fffe08 	b.hi	1700 <main+0x290>  // b.pmore
			max_ndims = X.ndims[m];
	}
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1744:	2a1403fb 	mov	w27, w20
    1748:	d37d7e80 	ubfiz	x0, x20, #3, #32
    174c:	8b00027a 	add	x26, x19, x0
    1750:	b94077e2 	ldr	w2, [sp, #116]
    1754:	f90037e0 	str	x0, [sp, #104]
    1758:	2a1803e1 	mov	w1, w24
    175c:	f87b7a60 	ldr	x0, [x19, x27, lsl #3]
    1760:	94000688 	bl	3180 <sptNewMatrix>
    1764:	35001440 	cbnz	w0, 19ec <main+0x57c>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1768:	0f000400 	movi	v0.2s, #0x0
    176c:	f87b7a60 	ldr	x0, [x19, x27, lsl #3]
    1770:	940006fc 	bl	3360 <sptConstantMatrix>
    1774:	350013c0 	cbnz	w0, 19ec <main+0x57c>
	sptIndex stride = U[0]->stride;

	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    1778:	d37ef760 	lsl	x0, x27, #2
    177c:	97fffec1 	bl	1280 <malloc@plt>
    1780:	aa0003f8 	mov	x24, x0
	mats_order[0] = mode;
    1784:	b94073e3 	ldr	w3, [sp, #112]
	for(sptIndex i=1; i<nmodes; ++i)
    1788:	d2800020 	mov	x0, #0x1                   	// #1
	mats_order[0] = mode;
    178c:	b9000303 	str	w3, [x24]
	for(sptIndex i=1; i<nmodes; ++i)
    1790:	7100069f 	cmp	w20, #0x1
    1794:	54000109 	b.ls	17b4 <main+0x344>  // b.plast
		mats_order[i] = (mode+i) % nmodes;
    1798:	0b000062 	add	w2, w3, w0
    179c:	1ad40841 	udiv	w1, w2, w20
    17a0:	1b148821 	msub	w1, w1, w20, w2
    17a4:	b8207b01 	str	w1, [x24, x0, lsl #2]
	for(sptIndex i=1; i<nmodes; ++i)
    17a8:	91000400 	add	x0, x0, #0x1
    17ac:	6b00029f 	cmp	w20, w0
    17b0:	54ffff48 	b.hi	1798 <main+0x328>  // b.pmore

	/* For warm-up caches, timing not included */
	if(dev_id == -2) {
    17b4:	b9407be0 	ldr	w0, [sp, #120]
    17b8:	3100081f 	cmn	w0, #0x2
    17bc:	54001780 	b.eq	1aac <main+0x63c>  // b.none
		nthreads = 1;
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
	} else if(dev_id == -1) {
    17c0:	3100041f 	cmn	w0, #0x1
    17c4:	54001800 	b.eq	1ac4 <main+0x654>  // b.none
    17c8:	910203f7 	add	x23, sp, #0x80
		nthreads = 1;
    17cc:	5280003c 	mov	w28, #0x1                   	// #1
#endif
	}


	sptTimer timer;
	sptNewTimer(&timer, 0);
    17d0:	aa1703e0 	mov	x0, x23
    17d4:	52800001 	mov	w1, #0x0                   	// #0
    17d8:	94000ba2 	bl	4660 <sptNewTimer>
	sptStartTimer(timer);
    17dc:	528000b7 	mov	w23, #0x5                   	// #5
    17e0:	f94043e0 	ldr	x0, [sp, #128]
    17e4:	94000bbf 	bl	46e0 <sptStartTimer>

	for(int it=0; it<niters; ++it) {
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    17e8:	0f000400 	movi	v0.2s, #0x0
    17ec:	f9400340 	ldr	x0, [x26]
    17f0:	940006dc 	bl	3360 <sptConstantMatrix>
    17f4:	35000fc0 	cbnz	w0, 19ec <main+0x57c>
		if(dev_id == -2) {
    17f8:	b9407be0 	ldr	w0, [sp, #120]
    17fc:	3100081f 	cmn	w0, #0x2
    1800:	54000ea0 	b.eq	19d4 <main+0x564>  // b.none
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
		} else if(dev_id == -1) {
    1804:	3100041f 	cmn	w0, #0x1
    1808:	54000f60 	b.eq	19f4 <main+0x584>  // b.none
	for(int it=0; it<niters; ++it) {
    180c:	710006f7 	subs	w23, w23, #0x1
    1810:	54fffec1 	b.ne	17e8 <main+0x378>  // b.any
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
#endif
		}
	}

	sptStopTimer(timer);
    1814:	f94043e0 	ldr	x0, [sp, #128]
    1818:	94000be2 	bl	47a0 <sptStopTimer>

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    181c:	f94043e0 	ldr	x0, [sp, #128]
    1820:	528000a1 	mov	w1, #0x5                   	// #5
    1824:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    1828:	91074042 	add	x2, x2, #0x1d0
    182c:	94000c3a 	bl	4914 <sptPrintAverageElapsedTime>
    1830:	1e604001 	fmov	d1, d0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1834:	b94077e5 	ldr	w5, [sp, #116]
    1838:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    183c:	f94053e0 	ldr	x0, [sp, #160]
    1840:	1e630282 	ucvtf	d2, w20
    1844:	1e6300a4 	ucvtf	d4, w5
    1848:	fd419423 	ldr	d3, [x1, #808]
    184c:	9e630000 	ucvtf	d0, x0
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1850:	9b000360 	madd	x0, x27, x0, x0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1854:	1e640842 	fmul	d2, d2, d4
    1858:	1e630800 	fmul	d0, d0, d3
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    185c:	d37ef400 	lsl	x0, x0, #2
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1860:	1e600842 	fmul	d2, d2, d0
    1864:	1e611840 	fdiv	d0, d2, d1
	for (sptIndex m=0; m<nmodes; ++m) {
    1868:	340015b4 	cbz	w20, 1b1c <main+0x6ac>
    186c:	51000681 	sub	w1, w20, #0x1
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1870:	f9404fe4 	ldr	x4, [sp, #152]
    1874:	71000c3f 	cmp	w1, #0x3
    1878:	540002e9 	b.ls	18d4 <main+0x464>  // b.plast
    187c:	53027e83 	lsr	w3, w20, #2
    1880:	91004082 	add	x2, x4, #0x10
    1884:	51000463 	sub	w3, w3, #0x1
    1888:	4e040ca6 	dup	v6.4s, w5
    188c:	4f000404 	movi	v4.4s, #0x0
    1890:	aa0403e1 	mov	x1, x4
    1894:	4f000485 	movi	v5.4s, #0x4
    1898:	8b235042 	add	x2, x2, w3, uxtw #4
    189c:	d503201f 	nop
    18a0:	3cc10422 	ldr	q2, [x1], #16
    18a4:	4ea69c42 	mul	v2.4s, v2.4s, v6.4s
    18a8:	2ea5c043 	umull	v3.2d, v2.2s, v5.2s
    18ac:	6ea58043 	umlal2	v3.2d, v2.4s, v5.4s
    18b0:	4ee38484 	add	v4.2d, v4.2d, v3.2d
	for (sptIndex m=0; m<nmodes; ++m) {
    18b4:	eb01005f 	cmp	x2, x1
    18b8:	54ffff41 	b.ne	18a0 <main+0x430>  // b.any
    18bc:	5ef1b884 	addp	d4, v4.2d
    18c0:	121e7697 	and	w23, w20, #0xfffffffc
    18c4:	4e083c81 	mov	x1, v4.d[0]
    18c8:	8b010000 	add	x0, x0, x1
    18cc:	f240069f 	tst	x20, #0x3
    18d0:	540002c0 	b.eq	1928 <main+0x4b8>  // b.none
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18d4:	b8775881 	ldr	w1, [x4, w23, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18d8:	110006e2 	add	w2, w23, #0x1
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18dc:	1b017ca1 	mul	w1, w5, w1
    18e0:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    18e4:	6b14005f 	cmp	w2, w20
    18e8:	54000202 	b.cs	1928 <main+0x4b8>  // b.hs, b.nlast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18ec:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18f0:	11000ae2 	add	w2, w23, #0x2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18f4:	1b017ca1 	mul	w1, w5, w1
    18f8:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    18fc:	6b14005f 	cmp	w2, w20
    1900:	54000142 	b.cs	1928 <main+0x4b8>  // b.hs, b.nlast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1904:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    1908:	11000ef7 	add	w23, w23, #0x3
		bytes += X.ndims[m] * R * sizeof(sptValue);
    190c:	1b017ca1 	mul	w1, w5, w1
    1910:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1914:	6b17029f 	cmp	w20, w23
    1918:	54000089 	b.ls	1928 <main+0x4b8>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    191c:	b8775881 	ldr	w1, [x4, w23, uxtw #2]
    1920:	1b017ca1 	mul	w1, w5, w1
    1924:	8b214800 	add	x0, x0, w1, uxtw #2
	}
	double gbw = (double)bytes / aver_time / 1e9;
    1928:	9e630002 	ucvtf	d2, x0
    192c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1930:	fd419403 	ldr	d3, [x0, #808]
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1934:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1938:	910a8000 	add	x0, x0, #0x2a0
	double gbw = (double)bytes / aver_time / 1e9;
    193c:	1e630842 	fmul	d2, d2, d3
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1940:	1e611841 	fdiv	d1, d2, d1
    1944:	97fffeb7 	bl	1420 <printf@plt>

	if(fo != NULL) {
    1948:	b4000116 	cbz	x22, 1968 <main+0x4f8>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    194c:	f94037e0 	ldr	x0, [sp, #104]
    1950:	aa1603e1 	mov	x1, x22
    1954:	f8606a60 	ldr	x0, [x19, x0]
    1958:	94000c2a 	bl	4a00 <sptDumpMatrix>
    195c:	35000480 	cbnz	w0, 19ec <main+0x57c>
		fclose(fo);
    1960:	aa1603e0 	mov	x0, x22
    1964:	97fffe3b 	bl	1250 <fclose@plt>
	}

	sptFreeTimer(timer);
    1968:	f94043e0 	ldr	x0, [sp, #128]
		bytes += X.ndims[m] * R * sizeof(sptValue);
    196c:	d2800016 	mov	x22, #0x0                   	// #0
	sptFreeTimer(timer);
    1970:	94000c08 	bl	4990 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1974:	d503201f 	nop
		sptFreeMatrix(U[m]);
    1978:	f8767a60 	ldr	x0, [x19, x22, lsl #3]
	for(sptIndex m=0; m<nmodes; ++m) {
    197c:	910006d6 	add	x22, x22, #0x1
		sptFreeMatrix(U[m]);
    1980:	9400068c 	bl	33b0 <sptFreeMatrix>
	for(sptIndex m=0; m<nmodes; ++m) {
    1984:	6b16029f 	cmp	w20, w22
    1988:	54ffff88 	b.hi	1978 <main+0x508>  // b.pmore
	}
	sptFreeSparseTensor(&X);
    198c:	aa1903e0 	mov	x0, x25
    1990:	94000198 	bl	1ff0 <sptFreeSparseTensor>
	free(mats_order);
    1994:	aa1803e0 	mov	x0, x24
    1998:	97fffe8a 	bl	13c0 <free@plt>
	sptFreeMatrix(U[nmodes]);
    199c:	f87b7a60 	ldr	x0, [x19, x27, lsl #3]
    19a0:	94000684 	bl	33b0 <sptFreeMatrix>
	free(U);
    19a4:	aa1303e0 	mov	x0, x19
    19a8:	97fffe86 	bl	13c0 <free@plt>

	if (!random){
    19ac:	34000355 	cbz	w21, 1a14 <main+0x5a4>
		if(fPtr1 != NULL) fclose(fPtr1);
		if(fPtr2 != NULL) fclose(fPtr2);

	}
	return 0;
}
    19b0:	52800000 	mov	w0, #0x0                   	// #0
    19b4:	a9407bfd 	ldp	x29, x30, [sp]
    19b8:	a94153f3 	ldp	x19, x20, [sp, #16]
    19bc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    19c0:	a94363f7 	ldp	x23, x24, [sp, #48]
    19c4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    19c8:	a94573fb 	ldp	x27, x28, [sp, #80]
    19cc:	913203ff 	add	sp, sp, #0xc80
    19d0:	d65f03c0 	ret
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    19d4:	b94073e3 	ldr	w3, [sp, #112]
    19d8:	aa1803e2 	mov	x2, x24
    19dc:	aa1303e1 	mov	x1, x19
    19e0:	aa1903e0 	mov	x0, x25
    19e4:	94000813 	bl	3a30 <sptMTTKRP>
    19e8:	34fff120 	cbz	w0, 180c <main+0x39c>
				sptAssert(fo != NULL);
    19ec:	12800000 	mov	w0, #0xffffffff            	// #-1
    19f0:	97fffdfc 	bl	11e0 <exit@plt>
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    19f4:	b94073e3 	ldr	w3, [sp, #112]
    19f8:	2a1c03e4 	mov	w4, w28
    19fc:	aa1803e2 	mov	x2, x24
    1a00:	aa1303e1 	mov	x1, x19
    1a04:	aa1903e0 	mov	x0, x25
    1a08:	94000a9b 	bl	4474 <sptOmpMTTKRP>
    1a0c:	34fff000 	cbz	w0, 180c <main+0x39c>
    1a10:	17fffff7 	b	19ec <main+0x57c>
		FILE* fPtr1 = fopen(fvname, "r");
    1a14:	9112c3f6 	add	x22, sp, #0x4b0
    1a18:	90000033 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    1a1c:	9107a273 	add	x19, x19, #0x1e8
    1a20:	aa1603e0 	mov	x0, x22
    1a24:	aa1303e1 	mov	x1, x19
		FILE* fPtr2 = fopen(foname, "r");
    1a28:	912263f5 	add	x21, sp, #0x898
		FILE* fPtr1 = fopen(fvname, "r");
    1a2c:	97fffe0d 	bl	1260 <fopen@plt>
    1a30:	aa0003f4 	mov	x20, x0
		FILE* fPtr2 = fopen(foname, "r");
    1a34:	aa1303e1 	mov	x1, x19
    1a38:	aa1503e0 	mov	x0, x21
    1a3c:	97fffe09 	bl	1260 <fopen@plt>
    1a40:	aa0003f3 	mov	x19, x0
		if (fPtr1 == NULL || fPtr2 == NULL) {
    1a44:	f100029f 	cmp	x20, #0x0
    1a48:	fa401804 	ccmp	x0, #0x0, #0x4, ne  // ne = any
    1a4c:	54000200 	b.eq	1a8c <main+0x61c>  // b.none
			int diff = compareFile(fPtr1, fPtr2);
    1a50:	aa0003e1 	mov	x1, x0
    1a54:	aa1403e0 	mov	x0, x20
    1a58:	940000ca 	bl	1d80 <compareFile>
			if (diff == 0) {
    1a5c:	35000820 	cbnz	w0, 1b60 <main+0x6f0>
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
    1a60:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1a64:	aa1603e2 	mov	x2, x22
    1a68:	aa1503e1 	mov	x1, x21
    1a6c:	91092000 	add	x0, x0, #0x248
    1a70:	97fffe6c 	bl	1420 <printf@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1a74:	aa1403e0 	mov	x0, x20
    1a78:	97fffdf6 	bl	1250 <fclose@plt>
		if(fPtr2 != NULL) fclose(fPtr2);
    1a7c:	b4fff9b3 	cbz	x19, 19b0 <main+0x540>
    1a80:	aa1303e0 	mov	x0, x19
    1a84:	97fffdf3 	bl	1250 <fclose@plt>
	return 0;
    1a88:	17ffffca 	b	19b0 <main+0x540>
			printf("\nUnable to open file.\n");
    1a8c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1a90:	9107c000 	add	x0, x0, #0x1f0
    1a94:	97fffe3b 	bl	1380 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    1a98:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1a9c:	91082000 	add	x0, x0, #0x208
    1aa0:	97fffe38 	bl	1380 <puts@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1aa4:	b4fffed4 	cbz	x20, 1a7c <main+0x60c>
    1aa8:	17fffff3 	b	1a74 <main+0x604>
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1aac:	aa1803e2 	mov	x2, x24
    1ab0:	aa1303e1 	mov	x1, x19
    1ab4:	aa1903e0 	mov	x0, x25
    1ab8:	940007de 	bl	3a30 <sptMTTKRP>
    1abc:	34ffe860 	cbz	w0, 17c8 <main+0x358>
    1ac0:	17ffffcb 	b	19ec <main+0x57c>
		#pragma omp parallel
    1ac4:	52800024 	mov	w4, #0x1                   	// #1
    1ac8:	910203f7 	add	x23, sp, #0x80
    1acc:	52800003 	mov	w3, #0x0                   	// #0
    1ad0:	52800002 	mov	w2, #0x0                   	// #0
    1ad4:	aa1703e1 	mov	x1, x23
    1ad8:	90000000 	adrp	x0, 1000 <g_last_error+0x1000>
    1adc:	91338000 	add	x0, x0, #0xce0
    1ae0:	b90083e4 	str	w4, [sp, #128]
    1ae4:	97fffe57 	bl	1440 <GOMP_parallel@plt>
    1ae8:	b94083fc 	ldr	w28, [sp, #128]
        printf("\nnthreads: %d\n", nthreads);
    1aec:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1af0:	91070000 	add	x0, x0, #0x1c0
    1af4:	2a1c03e1 	mov	w1, w28
    1af8:	97fffe4a 	bl	1420 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1afc:	b94073e3 	ldr	w3, [sp, #112]
    1b00:	2a1c03e4 	mov	w4, w28
    1b04:	aa1803e2 	mov	x2, x24
    1b08:	aa1303e1 	mov	x1, x19
    1b0c:	aa1903e0 	mov	x0, x25
    1b10:	94000a59 	bl	4474 <sptOmpMTTKRP>
    1b14:	34ffe5e0 	cbz	w0, 17d0 <main+0x360>
    1b18:	17ffffb5 	b	19ec <main+0x57c>
	double gbw = (double)bytes / aver_time / 1e9;
    1b1c:	9e630002 	ucvtf	d2, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1b20:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1b24:	910a8000 	add	x0, x0, #0x2a0
	double gbw = (double)bytes / aver_time / 1e9;
    1b28:	1e630842 	fmul	d2, d2, d3
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1b2c:	1e611841 	fdiv	d1, d2, d1
    1b30:	97fffe3c 	bl	1420 <printf@plt>
	if(fo != NULL) {
    1b34:	b4000116 	cbz	x22, 1b54 <main+0x6e4>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1b38:	f94037e0 	ldr	x0, [sp, #104]
    1b3c:	aa1603e1 	mov	x1, x22
    1b40:	f8606a60 	ldr	x0, [x19, x0]
    1b44:	94000baf 	bl	4a00 <sptDumpMatrix>
    1b48:	35fff520 	cbnz	w0, 19ec <main+0x57c>
		fclose(fo);
    1b4c:	aa1603e0 	mov	x0, x22
    1b50:	97fffdc0 	bl	1250 <fclose@plt>
	sptFreeTimer(timer);
    1b54:	f94043e0 	ldr	x0, [sp, #128]
    1b58:	94000b8e 	bl	4990 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1b5c:	17ffff8c 	b	198c <main+0x51c>
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1b60:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1b64:	9109c000 	add	x0, x0, #0x270
    1b68:	97fffe06 	bl	1380 <puts@plt>
    1b6c:	17ffffc2 	b	1a74 <main+0x604>
				print_usage(argv);
    1b70:	f9400280 	ldr	x0, [x20]
    1b74:	94000064 	bl	1d04 <print_usage.isra.0>
				exit(1);
    1b78:	52800020 	mov	w0, #0x1                   	// #1
    1b7c:	97fffd99 	bl	11e0 <exit@plt>

0000000000001b80 <init_have_lse_atomics>:
    1b80:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1b84:	d2800200 	mov	x0, #0x10                  	// #16
    1b88:	910003fd 	mov	x29, sp
    1b8c:	97fffdf5 	bl	1360 <__getauxval@plt>
    1b90:	53082000 	ubfx	w0, w0, #8, #1
    1b94:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1b98:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1b9c:	390aa420 	strb	w0, [x1, #681]
    1ba0:	d65f03c0 	ret
    1ba4:	d503201f 	nop
    1ba8:	d503201f 	nop
    1bac:	d503201f 	nop

0000000000001bb0 <set_fast_math>:
    1bb0:	52a02000 	mov	w0, #0x1000000             	// #16777216
    1bb4:	d51b4400 	msr	fpcr, x0
    1bb8:	d65f03c0 	ret

0000000000001bbc <_start>:
    1bbc:	d280001d 	mov	x29, #0x0                   	// #0
    1bc0:	d280001e 	mov	x30, #0x0                   	// #0
    1bc4:	aa0003e5 	mov	x5, x0
    1bc8:	f94003e1 	ldr	x1, [sp]
    1bcc:	910023e2 	add	x2, sp, #0x8
    1bd0:	910003e6 	mov	x6, sp
    1bd4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    1bd8:	f947ec00 	ldr	x0, [x0, #4056]
    1bdc:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xf98c>
    1be0:	f947e863 	ldr	x3, [x3, #4048]
    1be4:	b00000a4 	adrp	x4, 16000 <__FRAME_END__+0xf98c>
    1be8:	f947cc84 	ldr	x4, [x4, #3992]
    1bec:	97fffdad 	bl	12a0 <__libc_start_main@plt>
    1bf0:	97fffde0 	bl	1370 <abort@plt>

0000000000001bf4 <call_weak_fn>:
    1bf4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    1bf8:	f947e400 	ldr	x0, [x0, #4040]
    1bfc:	b4000040 	cbz	x0, 1c04 <call_weak_fn+0x10>
    1c00:	17fffdd4 	b	1350 <__gmon_start__@plt>
    1c04:	d65f03c0 	ret
    1c08:	d503201f 	nop
    1c0c:	d503201f 	nop

0000000000001c10 <deregister_tm_clones>:
    1c10:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1c14:	910aa000 	add	x0, x0, #0x2a8
    1c18:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1c1c:	910aa021 	add	x1, x1, #0x2a8
    1c20:	eb00003f 	cmp	x1, x0
    1c24:	540000c0 	b.eq	1c3c <deregister_tm_clones+0x2c>  // b.none
    1c28:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xf98c>
    1c2c:	f947d021 	ldr	x1, [x1, #4000]
    1c30:	b4000061 	cbz	x1, 1c3c <deregister_tm_clones+0x2c>
    1c34:	aa0103f0 	mov	x16, x1
    1c38:	d61f0200 	br	x16
    1c3c:	d65f03c0 	ret

0000000000001c40 <register_tm_clones>:
    1c40:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1c44:	910aa000 	add	x0, x0, #0x2a8
    1c48:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1c4c:	910aa021 	add	x1, x1, #0x2a8
    1c50:	cb000021 	sub	x1, x1, x0
    1c54:	d37ffc22 	lsr	x2, x1, #63
    1c58:	8b810c41 	add	x1, x2, x1, asr #3
    1c5c:	9341fc21 	asr	x1, x1, #1
    1c60:	b40000c1 	cbz	x1, 1c78 <register_tm_clones+0x38>
    1c64:	b00000a2 	adrp	x2, 16000 <__FRAME_END__+0xf98c>
    1c68:	f947f042 	ldr	x2, [x2, #4064]
    1c6c:	b4000062 	cbz	x2, 1c78 <register_tm_clones+0x38>
    1c70:	aa0203f0 	mov	x16, x2
    1c74:	d61f0200 	br	x16
    1c78:	d65f03c0 	ret
    1c7c:	d503201f 	nop

0000000000001c80 <__do_global_dtors_aux>:
    1c80:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1c84:	910003fd 	mov	x29, sp
    1c88:	f9000bf3 	str	x19, [sp, #16]
    1c8c:	d00000b3 	adrp	x19, 17000 <memcpy@GLIBC_2.17>
    1c90:	394aa260 	ldrb	w0, [x19, #680]
    1c94:	35000140 	cbnz	w0, 1cbc <__do_global_dtors_aux+0x3c>
    1c98:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    1c9c:	f947d400 	ldr	x0, [x0, #4008]
    1ca0:	b4000080 	cbz	x0, 1cb0 <__do_global_dtors_aux+0x30>
    1ca4:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1ca8:	f940ac00 	ldr	x0, [x0, #344]
    1cac:	97fffd51 	bl	11f0 <__cxa_finalize@plt>
    1cb0:	97ffffd8 	bl	1c10 <deregister_tm_clones>
    1cb4:	52800020 	mov	w0, #0x1                   	// #1
    1cb8:	390aa260 	strb	w0, [x19, #680]
    1cbc:	f9400bf3 	ldr	x19, [sp, #16]
    1cc0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1cc4:	d65f03c0 	ret
    1cc8:	d503201f 	nop
    1ccc:	d503201f 	nop

0000000000001cd0 <frame_dummy>:
    1cd0:	17ffffdc 	b	1c40 <register_tm_clones>
    1cd4:	d503201f 	nop
    1cd8:	d503201f 	nop
    1cdc:	d503201f 	nop

0000000000001ce0 <main._omp_fn.0>:
		#pragma omp parallel
    1ce0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1ce4:	910003fd 	mov	x29, sp
    1ce8:	f9000bf3 	str	x19, [sp, #16]
    1cec:	aa0003f3 	mov	x19, x0
            nthreads = omp_get_num_threads();
    1cf0:	97fffd8c 	bl	1320 <omp_get_num_threads@plt>
    1cf4:	b9000260 	str	w0, [x19]
		#pragma omp parallel
    1cf8:	f9400bf3 	ldr	x19, [sp, #16]
    1cfc:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1d00:	d65f03c0 	ret

0000000000001d04 <print_usage.isra.0>:
static void print_usage(char ** argv) {
    1d04:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
	printf("Usage: %s [options] \n\n", argv[0]);
    1d08:	aa0003e1 	mov	x1, x0
    1d0c:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP._omp_fn.0+0x180>
static void print_usage(char ** argv) {
    1d10:	910003fd 	mov	x29, sp
	printf("Usage: %s [options] \n\n", argv[0]);
    1d14:	913ba040 	add	x0, x2, #0xee8
    1d18:	97fffdc2 	bl	1420 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1d1c:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0x180>
    1d20:	913c0000 	add	x0, x0, #0xf00
    1d24:	97fffd97 	bl	1380 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    1d28:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0x180>
    1d2c:	913cc000 	add	x0, x0, #0xf30
    1d30:	97fffd94 	bl	1380 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1d34:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0x180>
    1d38:	913da000 	add	x0, x0, #0xf68
    1d3c:	97fffd91 	bl	1380 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    1d40:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0x180>
    1d44:	913f4000 	add	x0, x0, #0xfd0
    1d48:	97fffd8e 	bl	1380 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    1d4c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1d50:	91008000 	add	x0, x0, #0x20
    1d54:	97fffd8b 	bl	1380 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    1d58:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1d5c:	91018000 	add	x0, x0, #0x60
    1d60:	97fffd88 	bl	1380 <puts@plt>
	printf("         --help\n");
    1d64:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1d68:	9103c000 	add	x0, x0, #0xf0
    1d6c:	97fffd85 	bl	1380 <puts@plt>
}
    1d70:	a8c17bfd 	ldp	x29, x30, [sp], #16
	printf("\n");
    1d74:	52800140 	mov	w0, #0xa                   	// #10
    1d78:	17fffdb6 	b	1450 <putchar@plt>
    1d7c:	d503201f 	nop

0000000000001d80 <compareFile>:



int compareFile(FILE * fPtr1, FILE * fPtr2)
{
    1d80:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1d84:	910003fd 	mov	x29, sp
    1d88:	a90153f3 	stp	x19, x20, [sp, #16]
    1d8c:	aa0103f4 	mov	x20, x1
    1d90:	f90013f5 	str	x21, [sp, #32]
    1d94:	aa0003f5 	mov	x21, x0
    1d98:	14000006 	b	1db0 <compareFile+0x30>
		ch1 = fgetc(fPtr1);
		ch2 = fgetc(fPtr2);
		if (ch1 != ch2) {
			return -1;
		}
	} while (ch1 != EOF && ch2 != EOF);
    1d9c:	3100067f 	cmn	w19, #0x1
    1da0:	1a9f07e2 	cset	w2, ne  // ne = any
    1da4:	7100005f 	cmp	w2, #0x0
    1da8:	3a411a64 	ccmn	w19, #0x1, #0x4, ne  // ne = any
    1dac:	540001a0 	b.eq	1de0 <compareFile+0x60>  // b.none
		ch1 = fgetc(fPtr1);
    1db0:	aa1503e0 	mov	x0, x21
    1db4:	97fffd3f 	bl	12b0 <fgetc@plt>
    1db8:	2a0003f3 	mov	w19, w0
		ch2 = fgetc(fPtr2);
    1dbc:	aa1403e0 	mov	x0, x20
    1dc0:	97fffd3c 	bl	12b0 <fgetc@plt>
		if (ch1 != ch2) {
    1dc4:	6b00027f 	cmp	w19, w0
    1dc8:	54fffea0 	b.eq	1d9c <compareFile+0x1c>  // b.none
			return -1;
    1dcc:	12800000 	mov	w0, #0xffffffff            	// #-1
	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
		return 0;
	else
		return -1;
}
    1dd0:	a94153f3 	ldp	x19, x20, [sp, #16]
    1dd4:	f94013f5 	ldr	x21, [sp, #32]
    1dd8:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1ddc:	d65f03c0 	ret
	if (ch1 == EOF && ch2 == EOF)
    1de0:	4b0203e0 	neg	w0, w2
}
    1de4:	a94153f3 	ldp	x19, x20, [sp, #16]
    1de8:	f94013f5 	ldr	x21, [sp, #32]
    1dec:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1df0:	d65f03c0 	ret
    1df4:	d503201f 	nop
    1df8:	d503201f 	nop
    1dfc:	d503201f 	nop

0000000000001e00 <sptNewSparseTensor>:
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1e00:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    1e04:	910003fd 	mov	x29, sp
    1e08:	a9025bf5 	stp	x21, x22, [sp, #32]
    1e0c:	aa0003f6 	mov	x22, x0
	}
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
//	spt_CheckOSError(!tsr->ndims, "SpTns New");
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
	tsr->nnz = 0;
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1e10:	52800315 	mov	w21, #0x18                  	// #24
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1e14:	a90153f3 	stp	x19, x20, [sp, #16]
    1e18:	2a0103f3 	mov	w19, w1
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1e1c:	9bb57c35 	umull	x21, w1, w21
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1e20:	f9001bf7 	str	x23, [sp, #48]
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1e24:	d37e7e77 	ubfiz	x23, x19, #2, #32
	tsr->nmodes = nmodes;
    1e28:	b90002c1 	str	w1, [x22]
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1e2c:	aa0203f4 	mov	x20, x2
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1e30:	aa1703e0 	mov	x0, x23
    1e34:	97fffd13 	bl	1280 <malloc@plt>
    1e38:	f90006c0 	str	x0, [x22, #8]
	for(i = 0; i < nmodes; ++i) {
    1e3c:	34000a33 	cbz	w19, 1f80 <sptNewSparseTensor+0x180>
    1e40:	51000661 	sub	w1, w19, #0x1
    1e44:	7100143f 	cmp	w1, #0x5
    1e48:	54000ce9 	b.ls	1fe4 <sptNewSparseTensor+0x1e4>  // b.plast
    1e4c:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    1e50:	53027e64 	lsr	w4, w19, #2
    1e54:	4f000482 	movi	v2.4s, #0x4
    1e58:	aa0003e3 	mov	x3, x0
    1e5c:	3dc0dc20 	ldr	q0, [x1, #880]
    1e60:	8b245004 	add	x4, x0, w4, uxtw #4
    1e64:	d503201f 	nop
    1e68:	4ea01c01 	mov	v1.16b, v0.16b
    1e6c:	4ea28400 	add	v0.4s, v0.4s, v2.4s
		tsr->sortorder[i] = i;
    1e70:	3c810461 	str	q1, [x3], #16
	for(i = 0; i < nmodes; ++i) {
    1e74:	eb04007f 	cmp	x3, x4
    1e78:	54ffff81 	b.ne	1e68 <sptNewSparseTensor+0x68>  // b.any
    1e7c:	121e7663 	and	w3, w19, #0xfffffffc
    1e80:	f240067f 	tst	x19, #0x3
    1e84:	54000320 	b.eq	1ee8 <sptNewSparseTensor+0xe8>  // b.none
		tsr->sortorder[i] = i;
    1e88:	2a0303e1 	mov	w1, w3
	for(i = 0; i < nmodes; ++i) {
    1e8c:	11000462 	add	w2, w3, #0x1
		tsr->sortorder[i] = i;
    1e90:	b8217803 	str	w3, [x0, x1, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1e94:	6b02027f 	cmp	w19, w2
    1e98:	54000289 	b.ls	1ee8 <sptNewSparseTensor+0xe8>  // b.plast
    1e9c:	11000861 	add	w1, w3, #0x2
		tsr->sortorder[i] = i;
    1ea0:	b8225802 	str	w2, [x0, w2, uxtw #2]
	for(i = 0; i < nmodes; ++i) {
    1ea4:	6b01027f 	cmp	w19, w1
    1ea8:	54000209 	b.ls	1ee8 <sptNewSparseTensor+0xe8>  // b.plast
		tsr->sortorder[i] = i;
    1eac:	2a0103e4 	mov	w4, w1
	for(i = 0; i < nmodes; ++i) {
    1eb0:	11000c62 	add	w2, w3, #0x3
		tsr->sortorder[i] = i;
    1eb4:	b8247801 	str	w1, [x0, x4, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1eb8:	6b02027f 	cmp	w19, w2
    1ebc:	54000169 	b.ls	1ee8 <sptNewSparseTensor+0xe8>  // b.plast
    1ec0:	11001061 	add	w1, w3, #0x4
		tsr->sortorder[i] = i;
    1ec4:	b8225802 	str	w2, [x0, w2, uxtw #2]
	for(i = 0; i < nmodes; ++i) {
    1ec8:	6b01027f 	cmp	w19, w1
    1ecc:	540000e9 	b.ls	1ee8 <sptNewSparseTensor+0xe8>  // b.plast
		tsr->sortorder[i] = i;
    1ed0:	2a0103e2 	mov	w2, w1
	for(i = 0; i < nmodes; ++i) {
    1ed4:	11001463 	add	w3, w3, #0x5
		tsr->sortorder[i] = i;
    1ed8:	b8227801 	str	w1, [x0, x2, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1edc:	6b03027f 	cmp	w19, w3
    1ee0:	54000049 	b.ls	1ee8 <sptNewSparseTensor+0xe8>  // b.plast
		tsr->sortorder[i] = i;
    1ee4:	b8235803 	str	w3, [x0, w3, uxtw #2]
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1ee8:	aa1703e0 	mov	x0, x23
    1eec:	97fffce5 	bl	1280 <malloc@plt>
    1ef0:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1ef4:	aa1403e1 	mov	x1, x20
    1ef8:	aa1703e2 	mov	x2, x23
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1efc:	d2800014 	mov	x20, #0x0                   	// #0
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1f00:	97fffcb4 	bl	11d0 <memcpy@plt>
	tsr->nnz = 0;
    1f04:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1f08:	aa1503e0 	mov	x0, x21
    1f0c:	52800315 	mov	w21, #0x18                  	// #24
    1f10:	97fffcdc 	bl	1280 <malloc@plt>
    1f14:	f90012c0 	str	x0, [x22, #32]
//	spt_CheckOSError(!tsr->inds, "SpTns New");
	for(i = 0; i < nmodes; ++i) {
    1f18:	9bb57e75 	umull	x21, w19, w21
    1f1c:	14000004 	b	1f2c <sptNewSparseTensor+0x12c>
    1f20:	eb15029f 	cmp	x20, x21
    1f24:	54000420 	b.eq	1fa8 <sptNewSparseTensor+0x1a8>  // b.none
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1f28:	f94012c0 	ldr	x0, [x22, #32]
    1f2c:	8b140000 	add	x0, x0, x20
    1f30:	d2800002 	mov	x2, #0x0                   	// #0
    1f34:	d2800001 	mov	x1, #0x0                   	// #0
	for(i = 0; i < nmodes; ++i) {
    1f38:	91006294 	add	x20, x20, #0x18
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1f3c:	940000fd 	bl	2330 <sptNewIndexVector>
    1f40:	2a0003f3 	mov	w19, w0
		spt_CheckError(result, "SpTns New", NULL);
    1f44:	34fffee0 	cbz	w0, 1f20 <sptNewSparseTensor+0x120>
    1f48:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    1f4c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1f50:	910cc042 	add	x2, x2, #0x330
    1f54:	910d6000 	add	x0, x0, #0x358
    1f58:	2a1303e1 	mov	w1, w19
    1f5c:	d2800004 	mov	x4, #0x0                   	// #0
    1f60:	52800643 	mov	w3, #0x32                  	// #50
    1f64:	94000b0b 	bl	4b90 <spt_ComplainError>
	}
	result = sptNewValueVector(&tsr->values, 0, 0);
	spt_CheckError(result, "SpTns New", NULL);
	return 0;
}
    1f68:	2a1303e0 	mov	w0, w19
    1f6c:	a94153f3 	ldp	x19, x20, [sp, #16]
    1f70:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1f74:	f9401bf7 	ldr	x23, [sp, #48]
    1f78:	a8c47bfd 	ldp	x29, x30, [sp], #64
    1f7c:	d65f03c0 	ret
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1f80:	aa1703e0 	mov	x0, x23
    1f84:	97fffcbf 	bl	1280 <malloc@plt>
    1f88:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1f8c:	aa1703e2 	mov	x2, x23
    1f90:	aa1403e1 	mov	x1, x20
    1f94:	97fffc8f 	bl	11d0 <memcpy@plt>
	tsr->nnz = 0;
    1f98:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1f9c:	aa1503e0 	mov	x0, x21
    1fa0:	97fffcb8 	bl	1280 <malloc@plt>
    1fa4:	f90012c0 	str	x0, [x22, #32]
	result = sptNewValueVector(&tsr->values, 0, 0);
    1fa8:	9100a2c0 	add	x0, x22, #0x28
    1fac:	d2800002 	mov	x2, #0x0                   	// #0
    1fb0:	d2800001 	mov	x1, #0x0                   	// #0
    1fb4:	9400002f 	bl	2070 <sptNewValueVector>
    1fb8:	2a0003f3 	mov	w19, w0
	spt_CheckError(result, "SpTns New", NULL);
    1fbc:	34fffd60 	cbz	w0, 1f68 <sptNewSparseTensor+0x168>
    1fc0:	2a1303e1 	mov	w1, w19
    1fc4:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    1fc8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    1fcc:	910cc042 	add	x2, x2, #0x330
    1fd0:	910d6000 	add	x0, x0, #0x358
    1fd4:	d2800004 	mov	x4, #0x0                   	// #0
    1fd8:	528006a3 	mov	w3, #0x35                  	// #53
    1fdc:	94000aed 	bl	4b90 <spt_ComplainError>
    1fe0:	17ffffe2 	b	1f68 <sptNewSparseTensor+0x168>
	for(i = 0; i < nmodes; ++i) {
    1fe4:	52800003 	mov	w3, #0x0                   	// #0
    1fe8:	17ffffa8 	b	1e88 <sptNewSparseTensor+0x88>
    1fec:	d503201f 	nop

0000000000001ff0 <sptFreeSparseTensor>:

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    1ff0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1ff4:	910003fd 	mov	x29, sp
    1ff8:	a90153f3 	stp	x19, x20, [sp, #16]
    1ffc:	aa0003f4 	mov	x20, x0
	sptIndex i;
	for(i = 0; i < tsr->nmodes; ++i) {
    2000:	b9400000 	ldr	w0, [x0]
    2004:	340001a0 	cbz	w0, 2038 <sptFreeSparseTensor+0x48>
    2008:	52800013 	mov	w19, #0x0                   	// #0
    200c:	f90013f5 	str	x21, [sp, #32]
		sptFreeIndexVector(&tsr->inds[i]);
    2010:	52800315 	mov	w21, #0x18                  	// #24
    2014:	d503201f 	nop
    2018:	f9401280 	ldr	x0, [x20, #32]
    201c:	9bb50260 	umaddl	x0, w19, w21, x0
	for(i = 0; i < tsr->nmodes; ++i) {
    2020:	11000673 	add	w19, w19, #0x1
		sptFreeIndexVector(&tsr->inds[i]);
    2024:	94000150 	bl	2564 <sptFreeIndexVector>
	for(i = 0; i < tsr->nmodes; ++i) {
    2028:	b9400280 	ldr	w0, [x20]
    202c:	6b13001f 	cmp	w0, w19
    2030:	54ffff48 	b.hi	2018 <sptFreeSparseTensor+0x28>  // b.pmore
    2034:	f94013f5 	ldr	x21, [sp, #32]
	}
	free(tsr->sortorder);
    2038:	f9400680 	ldr	x0, [x20, #8]
    203c:	97fffce1 	bl	13c0 <free@plt>
	free(tsr->ndims);
    2040:	f9400a80 	ldr	x0, [x20, #16]
    2044:	97fffcdf 	bl	13c0 <free@plt>
	free(tsr->inds);
    2048:	f9401280 	ldr	x0, [x20, #32]
    204c:	97fffcdd 	bl	13c0 <free@plt>
	sptFreeValueVector(&tsr->values);
    2050:	9100a280 	add	x0, x20, #0x28
    2054:	940000b4 	bl	2324 <sptFreeValueVector>
	tsr->nmodes = 0;
    2058:	b900029f 	str	wzr, [x20]
}
    205c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2060:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2064:	d65f03c0 	ret
    2068:	d503201f 	nop
    206c:	d503201f 	nop

0000000000002070 <sptNewValueVector>:
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    2070:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2074:	f100083f 	cmp	x1, #0x2
    2078:	d2800043 	mov	x3, #0x2                   	// #2
    207c:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    2080:	910003fd 	mov	x29, sp
    2084:	eb02007f 	cmp	x3, x2
    2088:	aa0103e4 	mov	x4, x1
    208c:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
    2090:	a90153f3 	stp	x19, x20, [sp, #16]
    2094:	aa0003f3 	mov	x19, x0
	if(cap < 2) {
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    2098:	d2800021 	mov	x1, #0x1                   	// #1
	vec->cap = cap;
    209c:	a9000804 	stp	x4, x2, [x0]
	vec->data = malloc(cap * sizeof *vec->data);
    20a0:	d37ef440 	lsl	x0, x2, #2
    20a4:	97fffc8f 	bl	12e0 <calloc@plt>
    20a8:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "ValVec New");
    20ac:	b40000a0 	cbz	x0, 20c0 <sptNewValueVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    20b0:	52800000 	mov	w0, #0x0                   	// #0
}
    20b4:	a94153f3 	ldp	x19, x20, [sp, #16]
    20b8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    20bc:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "ValVec New");
    20c0:	97fffcdc 	bl	1430 <__errno_location@plt>
    20c4:	aa0003f3 	mov	x19, x0
	return 0;
    20c8:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "ValVec New");
    20cc:	b9400274 	ldr	w20, [x19]
    20d0:	3140429f 	cmn	w20, #0x10, lsl #12
    20d4:	54ffff00 	b.eq	20b4 <sptNewValueVector+0x44>  // b.none
    20d8:	2a1403e0 	mov	w0, w20
    20dc:	97fffc95 	bl	1330 <strerror@plt>
    20e0:	11404281 	add	w1, w20, #0x10, lsl #12
    20e4:	aa0003e4 	mov	x4, x0
    20e8:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    20ec:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    20f0:	910e0042 	add	x2, x2, #0x380
    20f4:	910ea000 	add	x0, x0, #0x3a8
    20f8:	528005c3 	mov	w3, #0x2e                  	// #46
    20fc:	94000aa5 	bl	4b90 <spt_ComplainError>
    2100:	b9400260 	ldr	w0, [x19]
    2104:	11404000 	add	w0, w0, #0x10, lsl #12
    2108:	17ffffeb 	b	20b4 <sptNewValueVector+0x44>
    210c:	d503201f 	nop

0000000000002110 <sptConstantValueVector>:
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2110:	f9400002 	ldr	x2, [x0]
    2114:	b4000302 	cbz	x2, 2174 <sptConstantValueVector+0x64>
    2118:	d1000441 	sub	x1, x2, #0x1
		vec->data[i] = val;
    211c:	f9400803 	ldr	x3, [x0, #16]
    2120:	f100083f 	cmp	x1, #0x2
    2124:	540002c9 	b.ls	217c <sptConstantValueVector+0x6c>  // b.plast
    2128:	d342fc41 	lsr	x1, x2, #2
    212c:	aa0303e0 	mov	x0, x3
    2130:	4e040401 	dup	v1.4s, v0.s[0]
    2134:	8b011061 	add	x1, x3, x1, lsl #4
    2138:	3c810401 	str	q1, [x0], #16
	for(sptNnzIndex i=0; i<vec->len; ++i)
    213c:	eb01001f 	cmp	x0, x1
    2140:	54ffffc1 	b.ne	2138 <sptConstantValueVector+0x28>  // b.any
    2144:	927ef440 	and	x0, x2, #0xfffffffffffffffc
    2148:	f240045f 	tst	x2, #0x3
    214c:	54000140 	b.eq	2174 <sptConstantValueVector+0x64>  // b.none
    2150:	91000401 	add	x1, x0, #0x1
		vec->data[i] = val;
    2154:	bc207860 	str	s0, [x3, x0, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2158:	eb02003f 	cmp	x1, x2
    215c:	540000c2 	b.cs	2174 <sptConstantValueVector+0x64>  // b.hs, b.nlast
    2160:	91000800 	add	x0, x0, #0x2
		vec->data[i] = val;
    2164:	bc217860 	str	s0, [x3, x1, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2168:	eb00005f 	cmp	x2, x0
    216c:	54000049 	b.ls	2174 <sptConstantValueVector+0x64>  // b.plast
		vec->data[i] = val;
    2170:	bc207860 	str	s0, [x3, x0, lsl #2]
	return 0;
}
    2174:	52800000 	mov	w0, #0x0                   	// #0
    2178:	d65f03c0 	ret
	for(sptNnzIndex i=0; i<vec->len; ++i)
    217c:	d2800000 	mov	x0, #0x0                   	// #0
    2180:	17fffff4 	b	2150 <sptConstantValueVector+0x40>

0000000000002184 <sptAppendValueVector>:
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    2184:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2188:	910003fd 	mov	x29, sp
    218c:	a90153f3 	stp	x19, x20, [sp, #16]
    2190:	aa0003f3 	mov	x19, x0
	if(vec->cap <= vec->len) {
    2194:	a9400001 	ldp	x1, x0, [x0]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2198:	f9400a74 	ldr	x20, [x19, #16]
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    219c:	fd001fe8 	str	d8, [sp, #56]
    21a0:	1e204008 	fmov	s8, s0
	if(vec->cap <= vec->len) {
    21a4:	eb01001f 	cmp	x0, x1
    21a8:	54000129 	b.ls	21cc <sptAppendValueVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "ValVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    21ac:	bc217a88 	str	s8, [x20, x1, lsl #2]
	++vec->len;
	return 0;
    21b0:	52800000 	mov	w0, #0x0                   	// #0
	++vec->len;
    21b4:	91000421 	add	x1, x1, #0x1
    21b8:	f9000261 	str	x1, [x19]
}
    21bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    21c0:	fd401fe8 	ldr	d8, [sp, #56]
    21c4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    21c8:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    21cc:	a9025bf5 	stp	x21, x22, [sp, #32]
    21d0:	8b400415 	add	x21, x0, x0, lsr #1
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    21d4:	aa1403e0 	mov	x0, x20
    21d8:	d37ef6a1 	lsl	x1, x21, #2
    21dc:	97fffc45 	bl	12f0 <realloc@plt>
    21e0:	aa0003f4 	mov	x20, x0
		spt_CheckOSError(!newdata, "ValVec Append");
    21e4:	b40000a0 	cbz	x0, 21f8 <sptAppendValueVector+0x74>
		vec->data = newdata;
    21e8:	a900d275 	stp	x21, x20, [x19, #8]
	vec->data[vec->len] = value;
    21ec:	f9400261 	ldr	x1, [x19]
		vec->data = newdata;
    21f0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    21f4:	17ffffee 	b	21ac <sptAppendValueVector+0x28>
		spt_CheckOSError(!newdata, "ValVec Append");
    21f8:	f9001bf7 	str	x23, [sp, #48]
    21fc:	97fffc8d 	bl	1430 <__errno_location@plt>
    2200:	b9400017 	ldr	w23, [x0]
    2204:	aa0003f6 	mov	x22, x0
    2208:	314042ff 	cmn	w23, #0x10, lsl #12
    220c:	54000061 	b.ne	2218 <sptAppendValueVector+0x94>  // b.any
    2210:	f9401bf7 	ldr	x23, [sp, #48]
    2214:	17fffff5 	b	21e8 <sptAppendValueVector+0x64>
    2218:	2a1703e0 	mov	w0, w23
    221c:	97fffc45 	bl	1330 <strerror@plt>
    2220:	114042e1 	add	w1, w23, #0x10, lsl #12
    2224:	aa0003e4 	mov	x4, x0
    2228:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    222c:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2230:	910e0042 	add	x2, x2, #0x380
    2234:	910ee000 	add	x0, x0, #0x3b8
    2238:	52800a43 	mov	w3, #0x52                  	// #82
    223c:	94000a55 	bl	4b90 <spt_ComplainError>
    2240:	b94002c0 	ldr	w0, [x22]
    2244:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2248:	11404000 	add	w0, w0, #0x10, lsl #12
    224c:	f9401bf7 	ldr	x23, [sp, #48]
    2250:	17ffffdb 	b	21bc <sptAppendValueVector+0x38>

0000000000002254 <sptResizeValueVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2254:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2258:	f100083f 	cmp	x1, #0x2
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    225c:	910003fd 	mov	x29, sp
    2260:	a90153f3 	stp	x19, x20, [sp, #16]
    2264:	aa0003f3 	mov	x19, x0
    2268:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    226c:	f9400400 	ldr	x0, [x0, #8]
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2270:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2274:	d2800055 	mov	x21, #0x2                   	// #2
    2278:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    227c:	eb15001f 	cmp	x0, x21
    2280:	540000e1 	b.ne	229c <sptResizeValueVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    2284:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    2288:	f9000261 	str	x1, [x19]
}
    228c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2290:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2294:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2298:	d65f03c0 	ret
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    229c:	f9400a60 	ldr	x0, [x19, #16]
    22a0:	d37ef6a1 	lsl	x1, x21, #2
    22a4:	97fffc13 	bl	12f0 <realloc@plt>
    22a8:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "ValVec Resize");
    22ac:	b4000100 	cbz	x0, 22cc <sptResizeValueVector+0x78>
		vec->cap = newcap;
    22b0:	a9005674 	stp	x20, x21, [x19]
	return 0;
    22b4:	52800000 	mov	w0, #0x0                   	// #0
		vec->data = newdata;
    22b8:	f9000a76 	str	x22, [x19, #16]
}
    22bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    22c0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    22c4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    22c8:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "ValVec Resize");
    22cc:	a90363f7 	stp	x23, x24, [sp, #48]
    22d0:	97fffc58 	bl	1430 <__errno_location@plt>
    22d4:	b9400018 	ldr	w24, [x0]
    22d8:	aa0003f7 	mov	x23, x0
    22dc:	3140431f 	cmn	w24, #0x10, lsl #12
    22e0:	54000061 	b.ne	22ec <sptResizeValueVector+0x98>  // b.any
    22e4:	a94363f7 	ldp	x23, x24, [sp, #48]
    22e8:	17fffff2 	b	22b0 <sptResizeValueVector+0x5c>
    22ec:	2a1803e0 	mov	w0, w24
    22f0:	97fffc10 	bl	1330 <strerror@plt>
    22f4:	11404301 	add	w1, w24, #0x10, lsl #12
    22f8:	aa0003e4 	mov	x4, x0
    22fc:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    2300:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2304:	910e0042 	add	x2, x2, #0x380
    2308:	910f2000 	add	x0, x0, #0x3c8
    230c:	52800d23 	mov	w3, #0x69                  	// #105
    2310:	94000a20 	bl	4b90 <spt_ComplainError>
    2314:	b94002e0 	ldr	w0, [x23]
    2318:	a94363f7 	ldp	x23, x24, [sp, #48]
    231c:	11404000 	add	w0, w0, #0x10, lsl #12
    2320:	17ffffdb 	b	228c <sptResizeValueVector+0x38>

0000000000002324 <sptFreeValueVector>:
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
	vec->len = 0;
    2324:	a9007c1f 	stp	xzr, xzr, [x0]
	vec->cap = 0;
	free(vec->data);
    2328:	f9400800 	ldr	x0, [x0, #16]
    232c:	17fffc25 	b	13c0 <free@plt>

0000000000002330 <sptNewIndexVector>:
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    2330:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2334:	f100083f 	cmp	x1, #0x2
    2338:	d2800043 	mov	x3, #0x2                   	// #2
    233c:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    2340:	910003fd 	mov	x29, sp
    2344:	eb02007f 	cmp	x3, x2
    2348:	aa0103e4 	mov	x4, x1
    234c:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
    2350:	a90153f3 	stp	x19, x20, [sp, #16]
    2354:	aa0003f3 	mov	x19, x0
	if(cap < 2) {
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    2358:	d2800021 	mov	x1, #0x1                   	// #1
	vec->cap = cap;
    235c:	a9000804 	stp	x4, x2, [x0]
	vec->data = malloc(cap * sizeof *vec->data);
    2360:	d37ef440 	lsl	x0, x2, #2
    2364:	97fffbdf 	bl	12e0 <calloc@plt>
    2368:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "IdxVec New");
    236c:	b40000a0 	cbz	x0, 2380 <sptNewIndexVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    2370:	52800000 	mov	w0, #0x0                   	// #0
}
    2374:	a94153f3 	ldp	x19, x20, [sp, #16]
    2378:	a8c27bfd 	ldp	x29, x30, [sp], #32
    237c:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "IdxVec New");
    2380:	97fffc2c 	bl	1430 <__errno_location@plt>
    2384:	aa0003f3 	mov	x19, x0
	return 0;
    2388:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "IdxVec New");
    238c:	b9400274 	ldr	w20, [x19]
    2390:	3140429f 	cmn	w20, #0x10, lsl #12
    2394:	54ffff00 	b.eq	2374 <sptNewIndexVector+0x44>  // b.none
    2398:	2a1403e0 	mov	w0, w20
    239c:	97fffbe5 	bl	1330 <strerror@plt>
    23a0:	11404281 	add	w1, w20, #0x10, lsl #12
    23a4:	aa0003e4 	mov	x4, x0
    23a8:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    23ac:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    23b0:	910e0042 	add	x2, x2, #0x380
    23b4:	910f6000 	add	x0, x0, #0x3d8
    23b8:	52801283 	mov	w3, #0x94                  	// #148
    23bc:	940009f5 	bl	4b90 <spt_ComplainError>
    23c0:	b9400260 	ldr	w0, [x19]
    23c4:	11404000 	add	w0, w0, #0x10, lsl #12
    23c8:	17ffffeb 	b	2374 <sptNewIndexVector+0x44>
    23cc:	d503201f 	nop

00000000000023d0 <sptAppendIndexVector>:
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    23d0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    23d4:	910003fd 	mov	x29, sp
    23d8:	a90153f3 	stp	x19, x20, [sp, #16]
    23dc:	aa0003f3 	mov	x19, x0
    23e0:	2a0103f4 	mov	w20, w1
	if(vec->cap <= vec->len) {
    23e4:	a9400002 	ldp	x2, x0, [x0]
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    23e8:	a9025bf5 	stp	x21, x22, [sp, #32]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    23ec:	f9400a75 	ldr	x21, [x19, #16]
	if(vec->cap <= vec->len) {
    23f0:	eb02001f 	cmp	x0, x2
    23f4:	54000129 	b.ls	2418 <sptAppendIndexVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "IdxVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    23f8:	b8227ab4 	str	w20, [x21, x2, lsl #2]
	++vec->len;
	return 0;
    23fc:	52800000 	mov	w0, #0x0                   	// #0
	++vec->len;
    2400:	91000442 	add	x2, x2, #0x1
    2404:	f9000262 	str	x2, [x19]
}
    2408:	a94153f3 	ldp	x19, x20, [sp, #16]
    240c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2410:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2414:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    2418:	8b400416 	add	x22, x0, x0, lsr #1
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    241c:	aa1503e0 	mov	x0, x21
    2420:	d37ef6c1 	lsl	x1, x22, #2
    2424:	97fffbb3 	bl	12f0 <realloc@plt>
    2428:	aa0003f5 	mov	x21, x0
		spt_CheckOSError(!newdata, "IdxVec Append");
    242c:	b4000080 	cbz	x0, 243c <sptAppendIndexVector+0x6c>
		vec->data = newdata;
    2430:	a900d676 	stp	x22, x21, [x19, #8]
	vec->data[vec->len] = value;
    2434:	f9400262 	ldr	x2, [x19]
    2438:	17fffff0 	b	23f8 <sptAppendIndexVector+0x28>
		spt_CheckOSError(!newdata, "IdxVec Append");
    243c:	a90363f7 	stp	x23, x24, [sp, #48]
    2440:	97fffbfc 	bl	1430 <__errno_location@plt>
    2444:	b9400018 	ldr	w24, [x0]
    2448:	aa0003f7 	mov	x23, x0
    244c:	3140431f 	cmn	w24, #0x10, lsl #12
    2450:	54000061 	b.ne	245c <sptAppendIndexVector+0x8c>  // b.any
    2454:	a94363f7 	ldp	x23, x24, [sp, #48]
    2458:	17fffff6 	b	2430 <sptAppendIndexVector+0x60>
    245c:	2a1803e0 	mov	w0, w24
    2460:	97fffbb4 	bl	1330 <strerror@plt>
    2464:	11404301 	add	w1, w24, #0x10, lsl #12
    2468:	aa0003e4 	mov	x4, x0
    246c:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    2470:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2474:	910e0042 	add	x2, x2, #0x380
    2478:	910fa000 	add	x0, x0, #0x3e8
    247c:	52801543 	mov	w3, #0xaa                  	// #170
    2480:	940009c4 	bl	4b90 <spt_ComplainError>
    2484:	b94002e0 	ldr	w0, [x23]
    2488:	a94363f7 	ldp	x23, x24, [sp, #48]
    248c:	11404000 	add	w0, w0, #0x10, lsl #12
    2490:	17ffffde 	b	2408 <sptAppendIndexVector+0x38>

0000000000002494 <sptResizeIndexVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    2494:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2498:	f100083f 	cmp	x1, #0x2
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    249c:	910003fd 	mov	x29, sp
    24a0:	a90153f3 	stp	x19, x20, [sp, #16]
    24a4:	aa0003f3 	mov	x19, x0
    24a8:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    24ac:	f9400400 	ldr	x0, [x0, #8]
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    24b0:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    24b4:	d2800055 	mov	x21, #0x2                   	// #2
    24b8:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    24bc:	eb15001f 	cmp	x0, x21
    24c0:	540000e1 	b.ne	24dc <sptResizeIndexVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    24c4:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    24c8:	f9000261 	str	x1, [x19]
}
    24cc:	a94153f3 	ldp	x19, x20, [sp, #16]
    24d0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    24d4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    24d8:	d65f03c0 	ret
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    24dc:	f9400a60 	ldr	x0, [x19, #16]
    24e0:	d37ef6a1 	lsl	x1, x21, #2
    24e4:	97fffb83 	bl	12f0 <realloc@plt>
    24e8:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "IdxVec Resize");
    24ec:	b4000100 	cbz	x0, 250c <sptResizeIndexVector+0x78>
		vec->cap = newcap;
    24f0:	a9005674 	stp	x20, x21, [x19]
	return 0;
    24f4:	52800000 	mov	w0, #0x0                   	// #0
		vec->data = newdata;
    24f8:	f9000a76 	str	x22, [x19, #16]
}
    24fc:	a94153f3 	ldp	x19, x20, [sp, #16]
    2500:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2504:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2508:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "IdxVec Resize");
    250c:	a90363f7 	stp	x23, x24, [sp, #48]
    2510:	97fffbc8 	bl	1430 <__errno_location@plt>
    2514:	b9400018 	ldr	w24, [x0]
    2518:	aa0003f7 	mov	x23, x0
    251c:	3140431f 	cmn	w24, #0x10, lsl #12
    2520:	54000061 	b.ne	252c <sptResizeIndexVector+0x98>  // b.any
    2524:	a94363f7 	ldp	x23, x24, [sp, #48]
    2528:	17fffff2 	b	24f0 <sptResizeIndexVector+0x5c>
    252c:	2a1803e0 	mov	w0, w24
    2530:	97fffb80 	bl	1330 <strerror@plt>
    2534:	11404301 	add	w1, w24, #0x10, lsl #12
    2538:	aa0003e4 	mov	x4, x0
    253c:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    2540:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2544:	910e0042 	add	x2, x2, #0x380
    2548:	910fe000 	add	x0, x0, #0x3f8
    254c:	52801823 	mov	w3, #0xc1                  	// #193
    2550:	94000990 	bl	4b90 <spt_ComplainError>
    2554:	b94002e0 	ldr	w0, [x23]
    2558:	a94363f7 	ldp	x23, x24, [sp, #48]
    255c:	11404000 	add	w0, w0, #0x10, lsl #12
    2560:	17ffffdb 	b	24cc <sptResizeIndexVector+0x38>

0000000000002564 <sptFreeIndexVector>:
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    2564:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2568:	910003fd 	mov	x29, sp
    256c:	f9000bf3 	str	x19, [sp, #16]
    2570:	aa0003f3 	mov	x19, x0
	free(vec->data);
    2574:	f9400800 	ldr	x0, [x0, #16]
    2578:	97fffb92 	bl	13c0 <free@plt>
	vec->len = 0;
    257c:	a9007e7f 	stp	xzr, xzr, [x19]
	vec->cap = 0;
}
    2580:	f9400bf3 	ldr	x19, [sp, #16]
    2584:	a8c27bfd 	ldp	x29, x30, [sp], #32
    2588:	d65f03c0 	ret
    258c:	d503201f 	nop

0000000000002590 <fill_binary_nnzidx._omp_fn.0>:
		sptIndex const BUF_LEN = 1024*1024;
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    2590:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2594:	910003fd 	mov	x29, sp
    2598:	f90013f5 	str	x21, [sp, #32]
    259c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    25a0:	34000355 	cbz	w21, 2608 <fill_binary_nnzidx._omp_fn.0+0x78>
    25a4:	a90153f3 	stp	x19, x20, [sp, #16]
    25a8:	aa0003f4 	mov	x20, x0
    25ac:	97fffb5d 	bl	1320 <omp_get_num_threads@plt>
    25b0:	2a0003f3 	mov	w19, w0
    25b4:	97fffb23 	bl	1240 <omp_get_thread_num@plt>
    25b8:	1ad30aa2 	udiv	w2, w21, w19
    25bc:	1b13d441 	msub	w1, w2, w19, w21
    25c0:	6b01001f 	cmp	w0, w1
    25c4:	54000283 	b.cc	2614 <fill_binary_nnzidx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    25c8:	1b000441 	madd	w1, w2, w0, w1
    25cc:	0b010043 	add	w3, w2, w1
    25d0:	6b03003f 	cmp	w1, w3
    25d4:	54000182 	b.cs	2604 <fill_binary_nnzidx._omp_fn.0+0x74>  // b.hs, b.nlast
    25d8:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    25dc:	b9401284 	ldr	w4, [x20, #16]
    25e0:	0b010080 	add	w0, w4, w1
    25e4:	0b030083 	add	w3, w4, w3
    25e8:	8b214842 	add	x2, x2, w1, uxtw #2
    25ec:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    25f0:	b8404441 	ldr	w1, [x2], #4
    25f4:	f82058a1 	str	x1, [x5, w0, uxtw #3]
    25f8:	11000400 	add	w0, w0, #0x1
    25fc:	6b00007f 	cmp	w3, w0
    2600:	54ffff81 	b.ne	25f0 <fill_binary_nnzidx._omp_fn.0+0x60>  // b.any
    2604:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    2608:	f94013f5 	ldr	x21, [sp, #32]
    260c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2610:	d65f03c0 	ret
    2614:	11000442 	add	w2, w2, #0x1
    2618:	52800001 	mov	w1, #0x0                   	// #0
    261c:	17ffffeb 	b	25c8 <fill_binary_nnzidx._omp_fn.0+0x38>

0000000000002620 <fill_binary_idx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    2620:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2624:	910003fd 	mov	x29, sp
    2628:	f90013f5 	str	x21, [sp, #32]
    262c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2630:	34000355 	cbz	w21, 2698 <fill_binary_idx._omp_fn.0+0x78>
    2634:	a90153f3 	stp	x19, x20, [sp, #16]
    2638:	aa0003f4 	mov	x20, x0
    263c:	97fffb39 	bl	1320 <omp_get_num_threads@plt>
    2640:	2a0003f3 	mov	w19, w0
    2644:	97fffaff 	bl	1240 <omp_get_thread_num@plt>
    2648:	1ad30aa2 	udiv	w2, w21, w19
    264c:	1b13d441 	msub	w1, w2, w19, w21
    2650:	6b01001f 	cmp	w0, w1
    2654:	54000283 	b.cc	26a4 <fill_binary_idx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2658:	1b000441 	madd	w1, w2, w0, w1
    265c:	0b010043 	add	w3, w2, w1
    2660:	6b03003f 	cmp	w1, w3
    2664:	54000182 	b.cs	2694 <fill_binary_idx._omp_fn.0+0x74>  // b.hs, b.nlast
    2668:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    266c:	b9401284 	ldr	w4, [x20, #16]
    2670:	0b010080 	add	w0, w4, w1
    2674:	0b030083 	add	w3, w4, w3
    2678:	8b214842 	add	x2, x2, w1, uxtw #2
    267c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2680:	b8404441 	ldr	w1, [x2], #4
    2684:	b82058a1 	str	w1, [x5, w0, uxtw #2]
    2688:	11000400 	add	w0, w0, #0x1
    268c:	6b00007f 	cmp	w3, w0
    2690:	54ffff81 	b.ne	2680 <fill_binary_idx._omp_fn.0+0x60>  // b.any
    2694:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    2698:	f94013f5 	ldr	x21, [sp, #32]
    269c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    26a0:	d65f03c0 	ret
    26a4:	11000442 	add	w2, w2, #0x1
    26a8:	52800001 	mov	w1, #0x0                   	// #0
    26ac:	17ffffeb 	b	2658 <fill_binary_idx._omp_fn.0+0x38>

00000000000026b0 <fill_binary_val._omp_fn.0>:
#endif

		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    26b0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    26b4:	910003fd 	mov	x29, sp
    26b8:	f90013f5 	str	x21, [sp, #32]
    26bc:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    26c0:	34000375 	cbz	w21, 272c <fill_binary_val._omp_fn.0+0x7c>
    26c4:	a90153f3 	stp	x19, x20, [sp, #16]
    26c8:	aa0003f4 	mov	x20, x0
    26cc:	97fffb15 	bl	1320 <omp_get_num_threads@plt>
    26d0:	2a0003f3 	mov	w19, w0
    26d4:	97fffadb 	bl	1240 <omp_get_thread_num@plt>
    26d8:	1ad30aa2 	udiv	w2, w21, w19
    26dc:	1b13d441 	msub	w1, w2, w19, w21
    26e0:	6b01001f 	cmp	w0, w1
    26e4:	540002a3 	b.cc	2738 <fill_binary_val._omp_fn.0+0x88>  // b.lo, b.ul, b.last
    26e8:	1b000441 	madd	w1, w2, w0, w1
    26ec:	0b010043 	add	w3, w2, w1
    26f0:	6b03003f 	cmp	w1, w3
    26f4:	540001a2 	b.cs	2728 <fill_binary_val._omp_fn.0+0x78>  // b.hs, b.nlast
    26f8:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    26fc:	b9401284 	ldr	w4, [x20, #16]
    2700:	0b010080 	add	w0, w4, w1
    2704:	0b030083 	add	w3, w4, w3
    2708:	8b214c42 	add	x2, x2, w1, uxtw #3
    270c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2710:	fc408440 	ldr	d0, [x2], #8
    2714:	1e624000 	fcvt	s0, d0
    2718:	bc2058a0 	str	s0, [x5, w0, uxtw #2]
    271c:	11000400 	add	w0, w0, #0x1
    2720:	6b00007f 	cmp	w3, w0
    2724:	54ffff61 	b.ne	2710 <fill_binary_val._omp_fn.0+0x60>  // b.any
    2728:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    272c:	f94013f5 	ldr	x21, [sp, #32]
    2730:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2734:	d65f03c0 	ret
    2738:	11000442 	add	w2, w2, #0x1
    273c:	52800001 	mov	w1, #0x0                   	// #0
    2740:	17ffffea 	b	26e8 <fill_binary_val._omp_fn.0+0x38>

0000000000002744 <sptLoadSparseTensor>:
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname)
{
    2744:	a9b57bfd 	stp	x29, x30, [sp, #-176]!
    2748:	910003fd 	mov	x29, sp
    274c:	a90153f3 	stp	x19, x20, [sp, #16]
    2750:	aa0203f3 	mov	x19, x2
    2754:	aa0003f4 	mov	x20, x0
	FILE * fp = fopen(fname, "r");
    2758:	aa0203e0 	mov	x0, x2
{
    275c:	a90363f7 	stp	x23, x24, [sp, #48]
    2760:	2a0103f7 	mov	w23, w1
	FILE * fp = fopen(fname, "r");
    2764:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    2768:	9107a021 	add	x1, x1, #0x1e8
    276c:	97fffabd 	bl	1260 <fopen@plt>
	sptAssert(fp != NULL);
    2770:	a9025bf5 	stp	x21, x22, [sp, #32]
    2774:	b4004ea0 	cbz	x0, 3148 <sptLoadSparseTensor+0xa04>
	char const * const suffix = strrchr(fname, '.');
    2778:	aa0003f5 	mov	x21, x0
    277c:	528005c1 	mov	w1, #0x2e                  	// #46
    2780:	aa1303e0 	mov	x0, x19
    2784:	97fffaef 	bl	1340 <strrchr@plt>
    2788:	aa0003f6 	mov	x22, x0
	if(suffix == NULL) {
    278c:	b4000200 	cbz	x0, 27cc <sptLoadSparseTensor+0x88>
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    2790:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    2794:	91102021 	add	x1, x1, #0x408
    2798:	97fffb02 	bl	13a0 <strcmp@plt>
    279c:	34000a20 	cbz	w0, 28e0 <sptLoadSparseTensor+0x19c>
    27a0:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    27a4:	aa1603e0 	mov	x0, x22
    27a8:	91104021 	add	x1, x1, #0x410
    27ac:	97fffafd 	bl	13a0 <strcmp@plt>
    27b0:	34004140 	cbz	w0, 2fd8 <sptLoadSparseTensor+0x894>
    27b4:	aa1603e0 	mov	x0, x22
    27b8:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    27bc:	91106021 	add	x1, x1, #0x418
    27c0:	97fffaf8 	bl	13a0 <strcmp@plt>
	} while(file_extensions[++idx].extension != NULL);
    27c4:	d2800041 	mov	x1, #0x2                   	// #2
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    27c8:	340008e0 	cbz	w0, 28e4 <sptLoadSparseTensor+0x1a0>
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    27cc:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    27d0:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    27d4:	aa1303e2 	mov	x2, x19
    27d8:	91108021 	add	x1, x1, #0x420
    27dc:	f947d800 	ldr	x0, [x0, #4016]
    27e0:	f9400000 	ldr	x0, [x0]
    27e4:	97fffb1f 	bl	1460 <fprintf@plt>
	iores = fscanf(fp, "%u", &tsr->nmodes);
    27e8:	f0000018 	adrp	x24, 5000 <_IO_stdin_used+0x120>
    27ec:	aa1403e2 	mov	x2, x20
    27f0:	91052301 	add	x1, x24, #0x148
    27f4:	aa1503e0 	mov	x0, x21
    27f8:	97fffaa6 	bl	1290 <__isoc99_fscanf@plt>
	spt_CheckOSError(iores < 0, "SpTns Load");
    27fc:	37f83ac0 	tbnz	w0, #31, 2f54 <sptLoadSparseTensor+0x810>
	tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    2800:	b9400296 	ldr	w22, [x20]
    2804:	d2800021 	mov	x1, #0x1                   	// #1
    2808:	a9046bf9 	stp	x25, x26, [sp, #64]
    280c:	d37e7ed3 	ubfiz	x19, x22, #2, #32
    2810:	aa1303e0 	mov	x0, x19
    2814:	97fffab3 	bl	12e0 <calloc@plt>
    2818:	f9000680 	str	x0, [x20, #8]
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    281c:	b4003f00 	cbz	x0, 2ffc <sptLoadSparseTensor+0x8b8>
	tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    2820:	aa1303e0 	mov	x0, x19
    2824:	97fffa97 	bl	1280 <malloc@plt>
    2828:	f9000a80 	str	x0, [x20, #16]
    282c:	aa0003e2 	mov	x2, x0
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    2830:	b40040e0 	cbz	x0, 304c <sptLoadSparseTensor+0x908>
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2834:	91052319 	add	x25, x24, #0x148
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2838:	52800013 	mov	w19, #0x0                   	// #0
    283c:	350000f6 	cbnz	w22, 2858 <sptLoadSparseTensor+0x114>
    2840:	14000172 	b	2e08 <sptLoadSparseTensor+0x6c4>
    2844:	b9400296 	ldr	w22, [x20]
    2848:	11000673 	add	w19, w19, #0x1
    284c:	6b16027f 	cmp	w19, w22
    2850:	54001762 	b.cs	2b3c <sptLoadSparseTensor+0x3f8>  // b.hs, b.nlast
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2854:	f9400a82 	ldr	x2, [x20, #16]
    2858:	8b334842 	add	x2, x2, w19, uxtw #2
    285c:	aa1903e1 	mov	x1, x25
    2860:	aa1503e0 	mov	x0, x21
    2864:	97fffa8b 	bl	1290 <__isoc99_fscanf@plt>
		spt_CheckOSError(iores != 1, "SpTns Load");
    2868:	7100041f 	cmp	w0, #0x1
    286c:	54fffec0 	b.eq	2844 <sptLoadSparseTensor+0x100>  // b.none
    2870:	97fffaf0 	bl	1430 <__errno_location@plt>
    2874:	b940001a 	ldr	w26, [x0]
    2878:	aa0003f6 	mov	x22, x0
    287c:	3140435f 	cmn	w26, #0x10, lsl #12
    2880:	54fffe20 	b.eq	2844 <sptLoadSparseTensor+0x100>  // b.none
    2884:	2a1a03e0 	mov	w0, w26
    2888:	97fffaaa 	bl	1330 <strerror@plt>
    288c:	11404341 	add	w1, w26, #0x10, lsl #12
    2890:	aa0003e4 	mov	x4, x0
    2894:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    2898:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    289c:	9111e262 	add	x2, x19, #0x478
    28a0:	91128280 	add	x0, x20, #0x4a0
    28a4:	52800a63 	mov	w3, #0x53                  	// #83
    28a8:	940008ba 	bl	4b90 <spt_ComplainError>
    28ac:	b94002c0 	ldr	w0, [x22]
    28b0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    28b4:	11404000 	add	w0, w0, #0x10, lsl #12

	int iores;
	switch(get_file_type(fname)) {
		case 0:
			iores = p_tt_read_file(tsr, start_index, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
    28b8:	350017e0 	cbnz	w0, 2bb4 <sptLoadSparseTensor+0x470>
    28bc:	d503201f 	nop
			iores = p_tt_read_binary_file(tsr, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
			break;
	}

	fclose(fp);
    28c0:	aa1503e0 	mov	x0, x21
    28c4:	97fffa63 	bl	1250 <fclose@plt>

	return 0;
    28c8:	52800000 	mov	w0, #0x0                   	// #0
}
    28cc:	a94153f3 	ldp	x19, x20, [sp, #16]
    28d0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    28d4:	a94363f7 	ldp	x23, x24, [sp, #48]
    28d8:	a8cb7bfd 	ldp	x29, x30, [sp], #176
    28dc:	d65f03c0 	ret
	size_t idx = 0;
    28e0:	d2800001 	mov	x1, #0x0                   	// #0
			return file_extensions[idx].type;
    28e4:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    28e8:	91350000 	add	x0, x0, #0xd40
    28ec:	8b011000 	add	x0, x0, x1, lsl #4
    28f0:	b9400800 	ldr	w0, [x0, #8]
	switch(get_file_type(fname)) {
    28f4:	34fff7a0 	cbz	w0, 27e8 <sptLoadSparseTensor+0xa4>
    28f8:	7100041f 	cmp	w0, #0x1
    28fc:	54fffe21 	b.ne	28c0 <sptLoadSparseTensor+0x17c>  // b.any
	fread(&(header->magic), sizeof(header->magic), 1, fin);
    2900:	aa1503e3 	mov	x3, x21
    2904:	d2800022 	mov	x2, #0x1                   	// #1
    2908:	d2800081 	mov	x1, #0x4                   	// #4
    290c:	910203e0 	add	x0, sp, #0x80
    2910:	a9046bf9 	stp	x25, x26, [sp, #64]
    2914:	a90573fb 	stp	x27, x28, [sp, #80]
    2918:	97fffaa6 	bl	13b0 <fread@plt>
	fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
    291c:	aa1503e3 	mov	x3, x21
    2920:	d2800022 	mov	x2, #0x1                   	// #1
    2924:	d2800101 	mov	x1, #0x8                   	// #8
    2928:	910223e0 	add	x0, sp, #0x88
    292c:	97fffaa1 	bl	13b0 <fread@plt>
	fread(&(header->val_width), sizeof(header->val_width), 1, fin);
    2930:	d2800101 	mov	x1, #0x8                   	// #8
    2934:	aa1503e3 	mov	x3, x21
    2938:	910243e0 	add	x0, sp, #0x90
    293c:	d2800022 	mov	x2, #0x1                   	// #1
    2940:	97fffa9c 	bl	13b0 <fread@plt>
	if(header->idx_width > PASTA_INDEX_TYPEWIDTH / 8) {
    2944:	f94047e1 	ldr	x1, [sp, #136]
    2948:	f100103f 	cmp	x1, #0x4
    294c:	54004068 	b.hi	3158 <sptLoadSparseTensor+0xa14>  // b.pmore
	if(header->val_width > PASTA_VALUE_TYPEWIDTH / 8) {
    2950:	f9404be0 	ldr	x0, [sp, #144]
    2954:	f100101f 	cmp	x0, #0x4
    2958:	54002448 	b.hi	2de0 <sptLoadSparseTensor+0x69c>  // b.pmore
	sptIndex nmodes = 0;
    295c:	b90077ff 	str	wzr, [sp, #116]
	sptNnzIndex nnz = 0;
    2960:	f9003fff 	str	xzr, [sp, #120]
	if(header->idx_width == sizeof(sptIndex)) {
    2964:	f100103f 	cmp	x1, #0x4
    2968:	54002180 	b.eq	2d98 <sptLoadSparseTensor+0x654>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    296c:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2970:	97fffa44 	bl	1280 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2974:	aa1503e3 	mov	x3, x21
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2978:	aa0003f3 	mov	x19, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    297c:	d2800022 	mov	x2, #0x1                   	// #1
    2980:	d2800081 	mov	x1, #0x4                   	// #4
    2984:	97fffa8b 	bl	13b0 <fread@plt>
    2988:	9000001b 	adrp	x27, 2000 <sptFreeSparseTensor+0x10>
#pragma omp parallel for schedule(static)
    298c:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2990:	9101d3e4 	add	x4, sp, #0x74
    2994:	910263e1 	add	x1, sp, #0x98
    2998:	52800003 	mov	w3, #0x0                   	// #0
    299c:	fd42d400 	ldr	d0, [x0, #1448]
    29a0:	52800002 	mov	w2, #0x0                   	// #0
    29a4:	91188360 	add	x0, x27, #0x620
    29a8:	a909cfe4 	stp	x4, x19, [sp, #152]
    29ac:	fd0057e0 	str	d0, [sp, #168]
    29b0:	97fffaa4 	bl	1440 <GOMP_parallel@plt>
		free(ubuf);
    29b4:	aa1303e0 	mov	x0, x19
    29b8:	97fffa82 	bl	13c0 <free@plt>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    29bc:	b94077f7 	ldr	w23, [sp, #116]
    29c0:	2a1703f6 	mov	w22, w23
    29c4:	d37e7ee0 	ubfiz	x0, x23, #2, #32
    29c8:	97fffa2e 	bl	1280 <malloc@plt>
    29cc:	f94047f3 	ldr	x19, [sp, #136]
    29d0:	aa0003fa 	mov	x26, x0
	if(header->idx_width == sizeof(sptIndex)) {
    29d4:	f100127f 	cmp	x19, #0x4
    29d8:	54001f80 	b.eq	2dc8 <sptLoadSparseTensor+0x684>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    29dc:	d2a00800 	mov	x0, #0x400000              	// #4194304
    29e0:	97fffa28 	bl	1280 <malloc@plt>
    29e4:	aa0003f8 	mov	x24, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    29e8:	34000337 	cbz	w23, 2a4c <sptLoadSparseTensor+0x308>
    29ec:	9000001b 	adrp	x27, 2000 <sptFreeSparseTensor+0x10>
    29f0:	910263f9 	add	x25, sp, #0x98
    29f4:	9118837b 	add	x27, x27, #0x620
    29f8:	52800016 	mov	w22, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    29fc:	52a0021c 	mov	w28, #0x100000              	// #1048576
    2a00:	4b1602f3 	sub	w19, w23, w22
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a04:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2a08:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a0c:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2a10:	1a9c9273 	csel	w19, w19, w28, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a14:	aa1803e0 	mov	x0, x24
    2a18:	2a1303e2 	mov	w2, w19
    2a1c:	97fffa65 	bl	13b0 <fread@plt>
#pragma omp parallel for schedule(static)
    2a20:	a909e3fa 	stp	x26, x24, [sp, #152]
    2a24:	aa1903e1 	mov	x1, x25
    2a28:	aa1b03e0 	mov	x0, x27
    2a2c:	52800003 	mov	w3, #0x0                   	// #0
    2a30:	52800002 	mov	w2, #0x0                   	// #0
    2a34:	29154ff6 	stp	w22, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2a38:	114402d6 	add	w22, w22, #0x100, lsl #12
    2a3c:	97fffa81 	bl	1440 <GOMP_parallel@plt>
    2a40:	6b1602ff 	cmp	w23, w22
    2a44:	54fffde8 	b.hi	2a00 <sptLoadSparseTensor+0x2bc>  // b.pmore
    2a48:	f94047f3 	ldr	x19, [sp, #136]
		free(ubuf);
    2a4c:	aa1803e0 	mov	x0, x24
    2a50:	97fffa5c 	bl	13c0 <free@plt>
	if(header->idx_width == sizeof(sptNnzIndex)) {
    2a54:	f100227f 	cmp	x19, #0x8
    2a58:	54001940 	b.eq	2d80 <sptLoadSparseTensor+0x63c>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2a5c:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2a60:	97fffa08 	bl	1280 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a64:	aa1503e3 	mov	x3, x21
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2a68:	aa0003f3 	mov	x19, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a6c:	d2800022 	mov	x2, #0x1                   	// #1
    2a70:	d2800081 	mov	x1, #0x4                   	// #4
    2a74:	97fffa4f 	bl	13b0 <fread@plt>
#pragma omp parallel for schedule(static)
    2a78:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2a7c:	9101e3e4 	add	x4, sp, #0x78
    2a80:	910263e1 	add	x1, sp, #0x98
    2a84:	52800003 	mov	w3, #0x0                   	// #0
    2a88:	fd42d400 	ldr	d0, [x0, #1448]
    2a8c:	52800002 	mov	w2, #0x0                   	// #0
    2a90:	90000000 	adrp	x0, 2000 <sptFreeSparseTensor+0x10>
    2a94:	91164000 	add	x0, x0, #0x590
    2a98:	a909cfe4 	stp	x4, x19, [sp, #152]
    2a9c:	fd0057e0 	str	d0, [sp, #168]
    2aa0:	97fffa68 	bl	1440 <GOMP_parallel@plt>
		free(ubuf);
    2aa4:	aa1303e0 	mov	x0, x19
    2aa8:	97fffa46 	bl	13c0 <free@plt>
	sptNewSparseTensor(tsr, nmodes, dims);
    2aac:	b94077e1 	ldr	w1, [sp, #116]
    2ab0:	aa1a03e2 	mov	x2, x26
    2ab4:	aa1403e0 	mov	x0, x20
    2ab8:	97fffcd2 	bl	1e00 <sptNewSparseTensor>
	for(sptIndex m=0; m < nmodes; ++m) {
    2abc:	b94077e0 	ldr	w0, [sp, #116]
	tsr->nnz = nnz;
    2ac0:	f9403fe1 	ldr	x1, [sp, #120]
    2ac4:	f9000e81 	str	x1, [x20, #24]
	for(sptIndex m=0; m < nmodes; ++m) {
    2ac8:	34000960 	cbz	w0, 2bf4 <sptLoadSparseTensor+0x4b0>
    2acc:	52800013 	mov	w19, #0x0                   	// #0
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2ad0:	52800316 	mov	w22, #0x18                  	// #24
    2ad4:	14000006 	b	2aec <sptLoadSparseTensor+0x3a8>
	for(sptIndex m=0; m < nmodes; ++m) {
    2ad8:	b94077e0 	ldr	w0, [sp, #116]
    2adc:	11000673 	add	w19, w19, #0x1
	result = sptResizeValueVector(&tsr->values, nnz);
    2ae0:	f9403fe1 	ldr	x1, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2ae4:	6b00027f 	cmp	w19, w0
    2ae8:	54000862 	b.cs	2bf4 <sptLoadSparseTensor+0x4b0>  // b.hs, b.nlast
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2aec:	f9401280 	ldr	x0, [x20, #32]
    2af0:	9bb60260 	umaddl	x0, w19, w22, x0
    2af4:	97fffe68 	bl	2494 <sptResizeIndexVector>
		spt_CheckError(result, "SpTns Read", NULL);
    2af8:	34ffff00 	cbz	w0, 2ad8 <sptLoadSparseTensor+0x394>
    2afc:	2a0003e1 	mov	w1, w0
    2b00:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    2b04:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2b08:	9111e262 	add	x2, x19, #0x478
    2b0c:	91166000 	add	x0, x0, #0x598
    2b10:	d2800004 	mov	x4, #0x0                   	// #0
    2b14:	528020a3 	mov	w3, #0x105                 	// #261
    2b18:	9400081e 	bl	4b90 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2b1c:	97fffa45 	bl	1430 <__errno_location@plt>
    2b20:	b9400016 	ldr	w22, [x0]
    2b24:	aa0003f4 	mov	x20, x0
    2b28:	314042df 	cmn	w22, #0x10, lsl #12
    2b2c:	540023a1 	b.ne	2fa0 <sptLoadSparseTensor+0x85c>  // b.any
    2b30:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2b34:	a94573fb 	ldp	x27, x28, [sp, #80]
    2b38:	17ffff62 	b	28c0 <sptLoadSparseTensor+0x17c>
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2b3c:	52800300 	mov	w0, #0x18                  	// #24
	tsr->nnz = 0;
    2b40:	f9000e9f 	str	xzr, [x20, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2b44:	9ba07ec0 	umull	x0, w22, w0
    2b48:	97fff9ce 	bl	1280 <malloc@plt>
    2b4c:	f9001280 	str	x0, [x20, #32]
    2b50:	aa0003f9 	mov	x25, x0
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2b54:	b4002a80 	cbz	x0, 30a4 <sptLoadSparseTensor+0x960>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2b58:	52800013 	mov	w19, #0x0                   	// #0
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2b5c:	5280031a 	mov	w26, #0x18                  	// #24
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2b60:	350000f6 	cbnz	w22, 2b7c <sptLoadSparseTensor+0x438>
    2b64:	140000b3 	b	2e30 <sptLoadSparseTensor+0x6ec>
    2b68:	b9400280 	ldr	w0, [x20]
    2b6c:	11000673 	add	w19, w19, #0x1
    2b70:	6b00027f 	cmp	w19, w0
    2b74:	540015e2 	b.cs	2e30 <sptLoadSparseTensor+0x6ec>  // b.hs, b.nlast
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2b78:	f9401299 	ldr	x25, [x20, #32]
    2b7c:	9bba6660 	umaddl	x0, w19, w26, x25
    2b80:	d2800002 	mov	x2, #0x0                   	// #0
    2b84:	d2800001 	mov	x1, #0x0                   	// #0
    2b88:	97fffdea 	bl	2330 <sptNewIndexVector>
		spt_CheckError(retval, "SpTns Load", NULL);
    2b8c:	34fffee0 	cbz	w0, 2b68 <sptLoadSparseTensor+0x424>
    2b90:	2a0003e1 	mov	w1, w0
    2b94:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    2b98:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    2b9c:	9111e262 	add	x2, x19, #0x478
    2ba0:	91128280 	add	x0, x20, #0x4a0
    2ba4:	d2800004 	mov	x4, #0x0                   	// #0
    2ba8:	52800b43 	mov	w3, #0x5a                  	// #90
    2bac:	940007f9 	bl	4b90 <spt_ComplainError>
    2bb0:	a9446bf9 	ldp	x25, x26, [sp, #64]
			spt_CheckOSError(iores != 0, "SpTns Load");
    2bb4:	97fffa1f 	bl	1430 <__errno_location@plt>
    2bb8:	b9400017 	ldr	w23, [x0]
    2bbc:	aa0003f6 	mov	x22, x0
    2bc0:	314042ff 	cmn	w23, #0x10, lsl #12
    2bc4:	54ffe7e0 	b.eq	28c0 <sptLoadSparseTensor+0x17c>  // b.none
    2bc8:	2a1703e0 	mov	w0, w23
    2bcc:	97fff9d9 	bl	1330 <strerror@plt>
    2bd0:	9111e262 	add	x2, x19, #0x478
    2bd4:	aa0003e4 	mov	x4, x0
    2bd8:	114042e1 	add	w1, w23, #0x10, lsl #12
    2bdc:	91128280 	add	x0, x20, #0x4a0
    2be0:	52802483 	mov	w3, #0x124                 	// #292
    2be4:	940007eb 	bl	4b90 <spt_ComplainError>
    2be8:	b94002c0 	ldr	w0, [x22]
    2bec:	11404000 	add	w0, w0, #0x10, lsl #12
    2bf0:	17ffff37 	b	28cc <sptLoadSparseTensor+0x188>
	result = sptResizeValueVector(&tsr->values, nnz);
    2bf4:	9100a280 	add	x0, x20, #0x28
    2bf8:	97fffd97 	bl	2254 <sptResizeValueVector>
	spt_CheckError(result, "SpTns Read", NULL);
    2bfc:	35002800 	cbnz	w0, 30fc <sptLoadSparseTensor+0x9b8>
	for(sptIndex m=0; m < nmodes; ++m) {
    2c00:	b94077e0 	ldr	w0, [sp, #116]
    2c04:	b9006bff 	str	wzr, [sp, #104]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2c08:	f9403ff3 	ldr	x19, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2c0c:	34002740 	cbz	w0, 30f4 <sptLoadSparseTensor+0x9b0>
    2c10:	90000019 	adrp	x25, 2000 <sptFreeSparseTensor+0x10>
    2c14:	910263fa 	add	x26, sp, #0x98
    2c18:	91188339 	add	x25, x25, #0x620
    2c1c:	d503201f 	nop
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2c20:	b9406be2 	ldr	w2, [sp, #104]
    2c24:	52800303 	mov	w3, #0x18                  	// #24
    2c28:	f9401280 	ldr	x0, [x20, #32]
	if(header->idx_width == sizeof(sptIndex)) {
    2c2c:	f94047e1 	ldr	x1, [sp, #136]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2c30:	9ba30040 	umaddl	x0, w2, w3, x0
    2c34:	f9400817 	ldr	x23, [x0, #16]
	if(header->idx_width == sizeof(sptIndex)) {
    2c38:	f100103f 	cmp	x1, #0x4
    2c3c:	54000940 	b.eq	2d64 <sptLoadSparseTensor+0x620>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2c40:	d2a00800 	mov	x0, #0x400000              	// #4194304
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2c44:	2a1303fc 	mov	w28, w19
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2c48:	97fff98e 	bl	1280 <malloc@plt>
    2c4c:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2c50:	34000313 	cbz	w19, 2cb0 <sptLoadSparseTensor+0x56c>
    2c54:	5280001b 	mov	w27, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2c58:	52a00218 	mov	w24, #0x100000              	// #1048576
    2c5c:	d503201f 	nop
    2c60:	4b1b0393 	sub	w19, w28, w27
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2c64:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2c68:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2c6c:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2c70:	1a989273 	csel	w19, w19, w24, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2c74:	aa1603e0 	mov	x0, x22
    2c78:	2a1303e2 	mov	w2, w19
    2c7c:	97fff9cd 	bl	13b0 <fread@plt>
#pragma omp parallel for schedule(static)
    2c80:	a909dbf7 	stp	x23, x22, [sp, #152]
    2c84:	aa1a03e1 	mov	x1, x26
    2c88:	aa1903e0 	mov	x0, x25
    2c8c:	52800003 	mov	w3, #0x0                   	// #0
    2c90:	52800002 	mov	w2, #0x0                   	// #0
    2c94:	29154ffb 	stp	w27, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2c98:	1144037b 	add	w27, w27, #0x100, lsl #12
    2c9c:	97fff9e9 	bl	1440 <GOMP_parallel@plt>
    2ca0:	6b1b039f 	cmp	w28, w27
    2ca4:	54fffde8 	b.hi	2c60 <sptLoadSparseTensor+0x51c>  // b.pmore
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2ca8:	f9403ff3 	ldr	x19, [sp, #120]
    2cac:	2a1303fc 	mov	w28, w19
		free(ubuf);
    2cb0:	aa1603e0 	mov	x0, x22
    2cb4:	97fff9c3 	bl	13c0 <free@plt>
	for(sptIndex m=0; m < nmodes; ++m) {
    2cb8:	b9406be1 	ldr	w1, [sp, #104]
    2cbc:	b94077e0 	ldr	w0, [sp, #116]
    2cc0:	11000421 	add	w1, w1, #0x1
    2cc4:	b9006be1 	str	w1, [sp, #104]
    2cc8:	6b00003f 	cmp	w1, w0
    2ccc:	54fffaa3 	b.cc	2c20 <sptLoadSparseTensor+0x4dc>  // b.lo, b.ul, b.last
	if(header->val_width == sizeof(sptValue)) {
    2cd0:	f9404be1 	ldr	x1, [sp, #144]
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2cd4:	f9401e98 	ldr	x24, [x20, #56]
	if(header->val_width == sizeof(sptValue)) {
    2cd8:	f100103f 	cmp	x1, #0x4
    2cdc:	54001820 	b.eq	2fe0 <sptLoadSparseTensor+0x89c>  // b.none
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    2ce0:	d2a01000 	mov	x0, #0x800000              	// #8388608
    2ce4:	90000017 	adrp	x23, 2000 <sptFreeSparseTensor+0x10>
    2ce8:	97fff966 	bl	1280 <malloc@plt>
    2cec:	911ac2f7 	add	x23, x23, #0x6b0
    2cf0:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2cf4:	910263f9 	add	x25, sp, #0x98
    2cf8:	52800014 	mov	w20, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2cfc:	52a0021a 	mov	w26, #0x100000              	// #1048576
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2d00:	3400029c 	cbz	w28, 2d50 <sptLoadSparseTensor+0x60c>
    2d04:	d503201f 	nop
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2d08:	4b140393 	sub	w19, w28, w20
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2d0c:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2d10:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2d14:	d2800101 	mov	x1, #0x8                   	// #8
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2d18:	1a9a9273 	csel	w19, w19, w26, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2d1c:	aa1603e0 	mov	x0, x22
    2d20:	2a1303e2 	mov	w2, w19
    2d24:	97fff9a3 	bl	13b0 <fread@plt>
#pragma omp parallel for schedule(static)
    2d28:	a909dbf8 	stp	x24, x22, [sp, #152]
    2d2c:	aa1903e1 	mov	x1, x25
    2d30:	aa1703e0 	mov	x0, x23
    2d34:	52800003 	mov	w3, #0x0                   	// #0
    2d38:	52800002 	mov	w2, #0x0                   	// #0
    2d3c:	29154ff4 	stp	w20, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2d40:	11440294 	add	w20, w20, #0x100, lsl #12
    2d44:	97fff9bf 	bl	1440 <GOMP_parallel@plt>
    2d48:	6b1c029f 	cmp	w20, w28
    2d4c:	54fffde3 	b.cc	2d08 <sptLoadSparseTensor+0x5c4>  // b.lo, b.ul, b.last
		free(ubuf);
    2d50:	aa1603e0 	mov	x0, x22
    2d54:	97fff99b 	bl	13c0 <free@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2d58:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2d5c:	a94573fb 	ldp	x27, x28, [sp, #80]
    2d60:	17fffed8 	b	28c0 <sptLoadSparseTensor+0x17c>
		fread(buffer, sizeof(sptIndex), count, fin);
    2d64:	2a1303e2 	mov	w2, w19
    2d68:	aa1703e0 	mov	x0, x23
    2d6c:	aa1503e3 	mov	x3, x21
    2d70:	97fff990 	bl	13b0 <fread@plt>
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2d74:	f9403ff3 	ldr	x19, [sp, #120]
    2d78:	2a1303fc 	mov	w28, w19
    2d7c:	17ffffcf 	b	2cb8 <sptLoadSparseTensor+0x574>
		fread(buffer, sizeof(sptNnzIndex), count, fin);
    2d80:	aa1303e1 	mov	x1, x19
    2d84:	aa1503e3 	mov	x3, x21
    2d88:	9101e3e0 	add	x0, sp, #0x78
    2d8c:	d2800022 	mov	x2, #0x1                   	// #1
    2d90:	97fff988 	bl	13b0 <fread@plt>
    2d94:	17ffff46 	b	2aac <sptLoadSparseTensor+0x368>
		fread(buffer, sizeof(sptIndex), count, fin);
    2d98:	aa1503e3 	mov	x3, x21
    2d9c:	d2800022 	mov	x2, #0x1                   	// #1
    2da0:	9101d3e0 	add	x0, sp, #0x74
    2da4:	97fff983 	bl	13b0 <fread@plt>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    2da8:	b94077f7 	ldr	w23, [sp, #116]
    2dac:	2a1703f6 	mov	w22, w23
    2db0:	d37e7ee0 	ubfiz	x0, x23, #2, #32
    2db4:	97fff933 	bl	1280 <malloc@plt>
    2db8:	f94047f3 	ldr	x19, [sp, #136]
    2dbc:	aa0003fa 	mov	x26, x0
	if(header->idx_width == sizeof(sptIndex)) {
    2dc0:	f100127f 	cmp	x19, #0x4
    2dc4:	54ffe0c1 	b.ne	29dc <sptLoadSparseTensor+0x298>  // b.any
		fread(buffer, sizeof(sptIndex), count, fin);
    2dc8:	aa1303e1 	mov	x1, x19
    2dcc:	aa1603e2 	mov	x2, x22
    2dd0:	aa1503e3 	mov	x3, x21
    2dd4:	97fff977 	bl	13b0 <fread@plt>
    2dd8:	f94047f3 	ldr	x19, [sp, #136]
    2ddc:	17ffff1e 	b	2a54 <sptLoadSparseTensor+0x310>
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    2de0:	900000a1 	adrp	x1, 16000 <__FRAME_END__+0xf98c>
    2de4:	d37df003 	lsl	x3, x0, #3
    2de8:	aa0303e2 	mov	x2, x3
    2dec:	f947d820 	ldr	x0, [x1, #4016]
    2df0:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    2df4:	91148021 	add	x1, x1, #0x520
    2df8:	f9400000 	ldr	x0, [x0]
    2dfc:	97fff999 	bl	1460 <fprintf@plt>
    2e00:	f94047e1 	ldr	x1, [sp, #136]
    2e04:	17fffed6 	b	295c <sptLoadSparseTensor+0x218>
	tsr->nnz = 0;
    2e08:	f9000e9f 	str	xzr, [x20, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2e0c:	d2800000 	mov	x0, #0x0                   	// #0
    2e10:	97fff91c 	bl	1280 <malloc@plt>
    2e14:	f9001280 	str	x0, [x20, #32]
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2e18:	b50000c0 	cbnz	x0, 2e30 <sptLoadSparseTensor+0x6ec>
    2e1c:	97fff985 	bl	1430 <__errno_location@plt>
    2e20:	b9400013 	ldr	w19, [x0]
    2e24:	aa0003fa 	mov	x26, x0
    2e28:	3140427f 	cmn	w19, #0x10, lsl #12
    2e2c:	54001461 	b.ne	30b8 <sptLoadSparseTensor+0x974>  // b.any
	retval = sptNewValueVector(&tsr->values, 0, 0);
    2e30:	9100a29a 	add	x26, x20, #0x28
    2e34:	d2800002 	mov	x2, #0x0                   	// #0
    2e38:	aa1a03e0 	mov	x0, x26
    2e3c:	d2800001 	mov	x1, #0x0                   	// #0
    2e40:	910263f9 	add	x25, sp, #0x98
    2e44:	97fffc8b 	bl	2070 <sptNewValueVector>
	spt_CheckError(retval, "SpTns Load", NULL);
    2e48:	350016c0 	cbnz	w0, 3120 <sptLoadSparseTensor+0x9dc>
			iores = fscanf(fp, "%u", &index);
    2e4c:	91052318 	add	x24, x24, #0x148
    2e50:	a90573fb 	stp	x27, x28, [sp, #80]
			iores = fscanf(fp, "%lf", &value);
    2e54:	f000001b 	adrp	x27, 5000 <_IO_stdin_used+0x120>
    2e58:	9113237b 	add	x27, x27, #0x4c8
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2e5c:	52800316 	mov	w22, #0x18                  	// #24
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e60:	b9400280 	ldr	w0, [x20]
    2e64:	52800013 	mov	w19, #0x0                   	// #0
    2e68:	350001a0 	cbnz	w0, 2e9c <sptLoadSparseTensor+0x758>
    2e6c:	14000020 	b	2eec <sptLoadSparseTensor+0x7a8>
			if(index < start_index) {
    2e70:	b9409be1 	ldr	w1, [sp, #152]
    2e74:	6b0102ff 	cmp	w23, w1
    2e78:	54000568 	b.hi	2f24 <sptLoadSparseTensor+0x7e0>  // b.pmore
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2e7c:	f9401280 	ldr	x0, [x20, #32]
    2e80:	4b170021 	sub	w1, w1, w23
    2e84:	9bb60260 	umaddl	x0, w19, w22, x0
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e88:	11000673 	add	w19, w19, #0x1
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2e8c:	97fffd51 	bl	23d0 <sptAppendIndexVector>
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e90:	b9400280 	ldr	w0, [x20]
    2e94:	6b00027f 	cmp	w19, w0
    2e98:	540002a2 	b.cs	2eec <sptLoadSparseTensor+0x7a8>  // b.hs, b.nlast
			iores = fscanf(fp, "%u", &index);
    2e9c:	aa1903e2 	mov	x2, x25
    2ea0:	aa1803e1 	mov	x1, x24
    2ea4:	aa1503e0 	mov	x0, x21
    2ea8:	97fff8fa 	bl	1290 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2eac:	7100041f 	cmp	w0, #0x1
    2eb0:	54fffe00 	b.eq	2e70 <sptLoadSparseTensor+0x72c>  // b.none
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2eb4:	b9400280 	ldr	w0, [x20]
    2eb8:	34ffe3c0 	cbz	w0, 2b30 <sptLoadSparseTensor+0x3ec>
    2ebc:	a9418a83 	ldp	x3, x2, [x20, #24]
    2ec0:	51000400 	sub	w0, w0, #0x1
    2ec4:	52800304 	mov	w4, #0x18                  	// #24
    2ec8:	91006041 	add	x1, x2, #0x18
    2ecc:	9ba40400 	umaddl	x0, w0, w4, x1
    2ed0:	14000002 	b	2ed8 <sptLoadSparseTensor+0x794>
    2ed4:	91006021 	add	x1, x1, #0x18
		tsr->inds[mode].len = tsr->nnz;
    2ed8:	f9000043 	str	x3, [x2]
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2edc:	aa0103e2 	mov	x2, x1
    2ee0:	eb01001f 	cmp	x0, x1
    2ee4:	54ffff81 	b.ne	2ed4 <sptLoadSparseTensor+0x790>  // b.any
    2ee8:	17ffff12 	b	2b30 <sptLoadSparseTensor+0x3ec>
			iores = fscanf(fp, "%lf", &value);
    2eec:	aa1903e2 	mov	x2, x25
    2ef0:	aa1b03e1 	mov	x1, x27
    2ef4:	aa1503e0 	mov	x0, x21
    2ef8:	97fff8e6 	bl	1290 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2efc:	7100041f 	cmp	w0, #0x1
    2f00:	54fffda1 	b.ne	2eb4 <sptLoadSparseTensor+0x770>  // b.any
			sptAppendValueVector(&tsr->values, value);
    2f04:	fd404fe0 	ldr	d0, [sp, #152]
    2f08:	aa1a03e0 	mov	x0, x26
    2f0c:	1e624000 	fcvt	s0, d0
    2f10:	97fffc9d 	bl	2184 <sptAppendValueVector>
			++tsr->nnz;
    2f14:	f9400e80 	ldr	x0, [x20, #24]
    2f18:	91000400 	add	x0, x0, #0x1
    2f1c:	f9000e80 	str	x0, [x20, #24]
	while(retval == 0) {
    2f20:	17ffffd0 	b	2e60 <sptLoadSparseTensor+0x71c>
				spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
    2f24:	f0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    2f28:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    2f2c:	9112c084 	add	x4, x4, #0x4b0
    2f30:	9111e262 	add	x2, x19, #0x478
    2f34:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    2f38:	52800d03 	mov	w3, #0x68                  	// #104
    2f3c:	91128280 	add	x0, x20, #0x4a0
    2f40:	52800061 	mov	w1, #0x3                   	// #3
    2f44:	94000713 	bl	4b90 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2f48:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2f4c:	a94573fb 	ldp	x27, x28, [sp, #80]
    2f50:	17ffff19 	b	2bb4 <sptLoadSparseTensor+0x470>
	spt_CheckOSError(iores < 0, "SpTns Load");
    2f54:	97fff937 	bl	1430 <__errno_location@plt>
    2f58:	b9400013 	ldr	w19, [x0]
    2f5c:	aa0003f6 	mov	x22, x0
    2f60:	3140427f 	cmn	w19, #0x10, lsl #12
    2f64:	54ffc4e0 	b.eq	2800 <sptLoadSparseTensor+0xbc>  // b.none
    2f68:	2a1303e0 	mov	w0, w19
    2f6c:	97fff8f1 	bl	1330 <strerror@plt>
    2f70:	11404261 	add	w1, w19, #0x10, lsl #12
    2f74:	aa0003e4 	mov	x4, x0
    2f78:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    2f7c:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    2f80:	9111e262 	add	x2, x19, #0x478
    2f84:	91128280 	add	x0, x20, #0x4a0
    2f88:	52800943 	mov	w3, #0x4a                  	// #74
    2f8c:	94000701 	bl	4b90 <spt_ComplainError>
    2f90:	b94002c0 	ldr	w0, [x22]
    2f94:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2f98:	34ffc940 	cbz	w0, 28c0 <sptLoadSparseTensor+0x17c>
    2f9c:	17ffff06 	b	2bb4 <sptLoadSparseTensor+0x470>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2fa0:	2a1603e0 	mov	w0, w22
    2fa4:	97fff8e3 	bl	1330 <strerror@plt>
    2fa8:	9111e262 	add	x2, x19, #0x478
    2fac:	aa0003e4 	mov	x4, x0
    2fb0:	114042c1 	add	w1, w22, #0x10, lsl #12
    2fb4:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    2fb8:	52802503 	mov	w3, #0x128                 	// #296
    2fbc:	91128000 	add	x0, x0, #0x4a0
    2fc0:	940006f4 	bl	4b90 <spt_ComplainError>
    2fc4:	b9400280 	ldr	w0, [x20]
    2fc8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2fcc:	11404000 	add	w0, w0, #0x10, lsl #12
    2fd0:	a94573fb 	ldp	x27, x28, [sp, #80]
    2fd4:	17fffe3e 	b	28cc <sptLoadSparseTensor+0x188>
	} while(file_extensions[++idx].extension != NULL);
    2fd8:	d2800021 	mov	x1, #0x1                   	// #1
    2fdc:	17fffe42 	b	28e4 <sptLoadSparseTensor+0x1a0>
		fread(buffer, sizeof(sptValue), count, fin);
    2fe0:	2a1303e2 	mov	w2, w19
    2fe4:	aa1803e0 	mov	x0, x24
    2fe8:	aa1503e3 	mov	x3, x21
    2fec:	97fff8f1 	bl	13b0 <fread@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2ff0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2ff4:	a94573fb 	ldp	x27, x28, [sp, #80]
    2ff8:	17fffe32 	b	28c0 <sptLoadSparseTensor+0x17c>
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    2ffc:	97fff90d 	bl	1430 <__errno_location@plt>
    3000:	b940001a 	ldr	w26, [x0]
    3004:	aa0003f9 	mov	x25, x0
    3008:	3140435f 	cmn	w26, #0x10, lsl #12
    300c:	54ffc0a0 	b.eq	2820 <sptLoadSparseTensor+0xdc>  // b.none
    3010:	2a1a03e0 	mov	w0, w26
    3014:	97fff8c7 	bl	1330 <strerror@plt>
    3018:	11404341 	add	w1, w26, #0x10, lsl #12
    301c:	aa0003e4 	mov	x4, x0
    3020:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    3024:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    3028:	9111e262 	add	x2, x19, #0x478
    302c:	91128280 	add	x0, x20, #0x4a0
    3030:	528009a3 	mov	w3, #0x4d                  	// #77
    3034:	940006d7 	bl	4b90 <spt_ComplainError>
    3038:	b9400320 	ldr	w0, [x25]
    303c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3040:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    3044:	34ffc3e0 	cbz	w0, 28c0 <sptLoadSparseTensor+0x17c>
    3048:	17fffedb 	b	2bb4 <sptLoadSparseTensor+0x470>
    304c:	f90037e0 	str	x0, [sp, #104]
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    3050:	97fff8f8 	bl	1430 <__errno_location@plt>
    3054:	b9400013 	ldr	w19, [x0]
    3058:	aa0003f9 	mov	x25, x0
    305c:	f94037e2 	ldr	x2, [sp, #104]
    3060:	3140427f 	cmn	w19, #0x10, lsl #12
    3064:	54ffbe80 	b.eq	2834 <sptLoadSparseTensor+0xf0>  // b.none
    3068:	2a1303e0 	mov	w0, w19
    306c:	97fff8b1 	bl	1330 <strerror@plt>
    3070:	11404261 	add	w1, w19, #0x10, lsl #12
    3074:	aa0003e4 	mov	x4, x0
    3078:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    307c:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    3080:	9111e262 	add	x2, x19, #0x478
    3084:	91128280 	add	x0, x20, #0x4a0
    3088:	52800a03 	mov	w3, #0x50                  	// #80
    308c:	940006c1 	bl	4b90 <spt_ComplainError>
    3090:	b9400320 	ldr	w0, [x25]
    3094:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3098:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    309c:	34ffc120 	cbz	w0, 28c0 <sptLoadSparseTensor+0x17c>
    30a0:	17fffec5 	b	2bb4 <sptLoadSparseTensor+0x470>
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    30a4:	97fff8e3 	bl	1430 <__errno_location@plt>
    30a8:	b9400013 	ldr	w19, [x0]
    30ac:	aa0003fa 	mov	x26, x0
    30b0:	3140427f 	cmn	w19, #0x10, lsl #12
    30b4:	54ffd520 	b.eq	2b58 <sptLoadSparseTensor+0x414>  // b.none
    30b8:	2a1303e0 	mov	w0, w19
    30bc:	97fff89d 	bl	1330 <strerror@plt>
    30c0:	11404261 	add	w1, w19, #0x10, lsl #12
    30c4:	aa0003e4 	mov	x4, x0
    30c8:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    30cc:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    30d0:	9111e262 	add	x2, x19, #0x478
    30d4:	91128280 	add	x0, x20, #0x4a0
    30d8:	52800ae3 	mov	w3, #0x57                  	// #87
    30dc:	940006ad 	bl	4b90 <spt_ComplainError>
    30e0:	b9400340 	ldr	w0, [x26]
    30e4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    30e8:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    30ec:	34ffbea0 	cbz	w0, 28c0 <sptLoadSparseTensor+0x17c>
    30f0:	17fffeb1 	b	2bb4 <sptLoadSparseTensor+0x470>
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    30f4:	2a1303fc 	mov	w28, w19
    30f8:	17fffef6 	b	2cd0 <sptLoadSparseTensor+0x58c>
	spt_CheckError(result, "SpTns Read", NULL);
    30fc:	2a0003e1 	mov	w1, w0
    3100:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    3104:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3108:	9111e262 	add	x2, x19, #0x478
    310c:	91166000 	add	x0, x0, #0x598
    3110:	d2800004 	mov	x4, #0x0                   	// #0
    3114:	52802103 	mov	w3, #0x108                 	// #264
    3118:	9400069e 	bl	4b90 <spt_ComplainError>
    311c:	17fffe80 	b	2b1c <sptLoadSparseTensor+0x3d8>
	spt_CheckError(retval, "SpTns Load", NULL);
    3120:	2a0003e1 	mov	w1, w0
    3124:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x120>
    3128:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x120>
    312c:	9111e262 	add	x2, x19, #0x478
    3130:	91128280 	add	x0, x20, #0x4a0
    3134:	d2800004 	mov	x4, #0x0                   	// #0
    3138:	52800ba3 	mov	w3, #0x5d                  	// #93
    313c:	94000695 	bl	4b90 <spt_ComplainError>
    3140:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3144:	17fffe9c 	b	2bb4 <sptLoadSparseTensor+0x470>
	sptAssert(fp != NULL);
    3148:	12800000 	mov	w0, #0xffffffff            	// #-1
    314c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3150:	a90573fb 	stp	x27, x28, [sp, #80]
    3154:	97fff823 	bl	11e0 <exit@plt>
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    3158:	f0000080 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    315c:	d37df023 	lsl	x3, x1, #3
    3160:	aa0303e2 	mov	x2, x3
    3164:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    3168:	f947d800 	ldr	x0, [x0, #4016]
    316c:	91134021 	add	x1, x1, #0x4d0
    3170:	f9400000 	ldr	x0, [x0]
    3174:	97fff8bb 	bl	1460 <fprintf@plt>
		exit(-1);
    3178:	12800000 	mov	w0, #0xffffffff            	// #-1
    317c:	97fff819 	bl	11e0 <exit@plt>

0000000000003180 <sptNewMatrix>:
 */
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
	mtx->nrows = nrows;
	mtx->ncols = ncols;
	mtx->cap = nrows != 0 ? nrows : 1;
	mtx->stride = ((ncols-1)/8+1)*8;
    3180:	51000443 	sub	w3, w2, #0x1
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3184:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    3188:	7100003f 	cmp	w1, #0x0
	mtx->stride = ((ncols-1)/8+1)*8;
    318c:	121d7063 	and	w3, w3, #0xfffffff8
    3190:	1a9f1424 	csinc	w4, w1, wzr, ne  // ne = any
    3194:	11002063 	add	w3, w3, #0x8
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3198:	910003fd 	mov	x29, sp
    319c:	a90153f3 	stp	x19, x20, [sp, #16]
    31a0:	aa0003f3 	mov	x19, x0
#ifdef _ISOC11_SOURCE
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    31a4:	1b047c74 	mul	w20, w3, w4
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    31a8:	a9025bf5 	stp	x21, x22, [sp, #32]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    31ac:	d2800400 	mov	x0, #0x20                  	// #32
	mtx->ncols = ncols;
    31b0:	29000a61 	stp	w1, w2, [x19]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    31b4:	d37ef694 	lsl	x20, x20, #2
	mtx->stride = ((ncols-1)/8+1)*8;
    31b8:	29010e64 	stp	w4, w3, [x19, #8]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    31bc:	aa1403e1 	mov	x1, x20
    31c0:	97fff840 	bl	12c0 <aligned_alloc@plt>
    31c4:	f9000a60 	str	x0, [x19, #16]
    31c8:	aa0003f5 	mov	x21, x0
		}
	}
#else
	mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
	spt_CheckOSError(!mtx->values, "Mtx New");
    31cc:	b4000140 	cbz	x0, 31f4 <sptNewMatrix+0x74>
	memset(mtx->values, 0, mtx->cap * mtx->stride * sizeof (sptValue));
    31d0:	aa1403e2 	mov	x2, x20
    31d4:	aa1503e0 	mov	x0, x21
    31d8:	52800001 	mov	w1, #0x0                   	// #0
    31dc:	97fff83d 	bl	12d0 <memset@plt>
	return 0;
    31e0:	52800000 	mov	w0, #0x0                   	// #0
}
    31e4:	a94153f3 	ldp	x19, x20, [sp, #16]
    31e8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    31ec:	a8c37bfd 	ldp	x29, x30, [sp], #48
    31f0:	d65f03c0 	ret
	spt_CheckOSError(!mtx->values, "Mtx New");
    31f4:	97fff88f 	bl	1430 <__errno_location@plt>
    31f8:	b9400016 	ldr	w22, [x0]
    31fc:	aa0003f3 	mov	x19, x0
    3200:	314042df 	cmn	w22, #0x10, lsl #12
    3204:	54fffe60 	b.eq	31d0 <sptNewMatrix+0x50>  // b.none
    3208:	2a1603e0 	mov	w0, w22
    320c:	97fff849 	bl	1330 <strerror@plt>
    3210:	114042c1 	add	w1, w22, #0x10, lsl #12
    3214:	aa0003e4 	mov	x4, x0
    3218:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    321c:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3220:	9116c042 	add	x2, x2, #0x5b0
    3224:	91176000 	add	x0, x0, #0x5d8
    3228:	52800703 	mov	w3, #0x38                  	// #56
    322c:	94000659 	bl	4b90 <spt_ComplainError>
    3230:	b9400260 	ldr	w0, [x19]
    3234:	11404000 	add	w0, w0, #0x10, lsl #12
    3238:	17ffffeb 	b	31e4 <sptNewMatrix+0x64>
    323c:	d503201f 	nop

0000000000003240 <sptRandomizeMatrix>:
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, bool random) {
    3240:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    3244:	910003fd 	mov	x29, sp
    3248:	a90363f7 	stp	x23, x24, [sp, #48]
    324c:	aa0003f8 	mov	x24, x0
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3250:	b9400000 	ldr	w0, [x0]
    3254:	34000440 	cbz	w0, 32dc <sptRandomizeMatrix+0x9c>
    3258:	12001c37 	and	w23, w1, #0xff
    325c:	a9025bf5 	stp	x21, x22, [sp, #32]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
			if (random) {
				srand(time(NULL) + (rand()%100) + i + j);
    3260:	5290a3f6 	mov	w22, #0x851f                	// #34079
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3264:	b9400700 	ldr	w0, [x24, #4]
				srand(time(NULL) + (rand()%100) + i + j);
    3268:	72aa3d76 	movk	w22, #0x51eb, lsl #16
    326c:	52800c95 	mov	w21, #0x64                  	// #100
    3270:	a90153f3 	stp	x19, x20, [sp, #16]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3274:	52800014 	mov	w20, #0x0                   	// #0
    3278:	a9046bf9 	stp	x25, x26, [sp, #64]
    327c:	f9002bfb 	str	x27, [sp, #80]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3280:	34000260 	cbz	w0, 32cc <sptRandomizeMatrix+0x8c>
    3284:	35000357 	cbnz	w23, 32ec <sptRandomizeMatrix+0xac>
    3288:	11134a9a 	add	w26, w20, #0x4d2
    328c:	52800013 	mov	w19, #0x0                   	// #0
			} else {
				srand(1234 + i + j);
    3290:	0b130340 	add	w0, w26, w19
    3294:	97fff85b 	bl	1400 <srand@plt>
			}
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3298:	b9400f19 	ldr	w25, [x24, #12]
    329c:	f9400b1b 	ldr	x27, [x24, #16]
    32a0:	940006c8 	bl	4dc0 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32a4:	b9400700 	ldr	w0, [x24, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    32a8:	1b194e99 	madd	w25, w20, w25, w19
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32ac:	11000673 	add	w19, w19, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    32b0:	bc397b60 	str	s0, [x27, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32b4:	6b00027f 	cmp	w19, w0
    32b8:	54fffec3 	b.cc	3290 <sptRandomizeMatrix+0x50>  // b.lo, b.ul, b.last
	for(sptIndex i=0; i<mtx->nrows; ++i)
    32bc:	b9400301 	ldr	w1, [x24]
    32c0:	11000694 	add	w20, w20, #0x1
    32c4:	6b01029f 	cmp	w20, w1
    32c8:	54fffdc3 	b.cc	3280 <sptRandomizeMatrix+0x40>  // b.lo, b.ul, b.last
    32cc:	a94153f3 	ldp	x19, x20, [sp, #16]
    32d0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    32d4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    32d8:	f9402bfb 	ldr	x27, [sp, #80]
		}
	return 0;
}
    32dc:	52800000 	mov	w0, #0x0                   	// #0
    32e0:	a94363f7 	ldp	x23, x24, [sp, #48]
    32e4:	a8c67bfd 	ldp	x29, x30, [sp], #96
    32e8:	d65f03c0 	ret
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32ec:	52800013 	mov	w19, #0x0                   	// #0
				srand(time(NULL) + (rand()%100) + i + j);
    32f0:	d2800000 	mov	x0, #0x0                   	// #0
    32f4:	97fff7df 	bl	1270 <time@plt>
    32f8:	aa0003f9 	mov	x25, x0
    32fc:	97fff801 	bl	1300 <rand@plt>
    3300:	9b367c01 	smull	x1, w0, w22
    3304:	0b13029a 	add	w26, w20, w19
    3308:	9365fc21 	asr	x1, x1, #37
    330c:	4b807c21 	sub	w1, w1, w0, asr #31
    3310:	1b158021 	msub	w1, w1, w21, w0
    3314:	0b190020 	add	w0, w1, w25
    3318:	0b1a0000 	add	w0, w0, w26
    331c:	97fff839 	bl	1400 <srand@plt>
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3320:	b9400f19 	ldr	w25, [x24, #12]
    3324:	f9400b1a 	ldr	x26, [x24, #16]
    3328:	940006a6 	bl	4dc0 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    332c:	b9400700 	ldr	w0, [x24, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3330:	1b194e99 	madd	w25, w20, w25, w19
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3334:	11000673 	add	w19, w19, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3338:	bc397b40 	str	s0, [x26, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    333c:	6b13001f 	cmp	w0, w19
    3340:	54fffd88 	b.hi	32f0 <sptRandomizeMatrix+0xb0>  // b.pmore
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3344:	b9400301 	ldr	w1, [x24]
    3348:	11000694 	add	w20, w20, #0x1
    334c:	6b01029f 	cmp	w20, w1
    3350:	54fff983 	b.cc	3280 <sptRandomizeMatrix+0x40>  // b.lo, b.ul, b.last
    3354:	17ffffde 	b	32cc <sptRandomizeMatrix+0x8c>
    3358:	d503201f 	nop
    335c:	d503201f 	nop

0000000000003360 <sptConstantMatrix>:
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3360:	b9400004 	ldr	w4, [x0]
    3364:	34000204 	cbz	w4, 33a4 <sptConstantMatrix+0x44>
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3368:	b9400405 	ldr	w5, [x0, #4]
    336c:	340001c5 	cbz	w5, 33a4 <sptConstantMatrix+0x44>
			mtx->values[i * mtx->stride + j] = val;
    3370:	b9400c06 	ldr	w6, [x0, #12]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3374:	52800003 	mov	w3, #0x0                   	// #0
			mtx->values[i * mtx->stride + j] = val;
    3378:	f9400802 	ldr	x2, [x0, #16]
    337c:	d503201f 	nop
    3380:	1b037cc0 	mul	w0, w6, w3
    3384:	0b050001 	add	w1, w0, w5
    3388:	bc205840 	str	s0, [x2, w0, uxtw #2]
		for(sptIndex j=0; j<mtx->ncols; ++j)
    338c:	11000400 	add	w0, w0, #0x1
    3390:	6b00003f 	cmp	w1, w0
    3394:	54ffffa1 	b.ne	3388 <sptConstantMatrix+0x28>  // b.any
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3398:	11000463 	add	w3, w3, #0x1
    339c:	6b04007f 	cmp	w3, w4
    33a0:	54ffff01 	b.ne	3380 <sptConstantMatrix+0x20>  // b.any
	return 0;
}
    33a4:	52800000 	mov	w0, #0x0                   	// #0
    33a8:	d65f03c0 	ret
    33ac:	d503201f 	nop

00000000000033b0 <sptFreeMatrix>:
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    33b0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    33b4:	910003fd 	mov	x29, sp
    33b8:	f9000bf3 	str	x19, [sp, #16]
    33bc:	aa0003f3 	mov	x19, x0
	free(mtx->values);
    33c0:	f9400800 	ldr	x0, [x0, #16]
    33c4:	97fff7ff 	bl	13c0 <free@plt>
	mtx->nrows = 0;
    33c8:	a9007e7f 	stp	xzr, xzr, [x19]
	mtx->ncols = 0;
	mtx->cap = 0;
	mtx->stride = 0;
}
    33cc:	f9400bf3 	ldr	x19, [sp, #16]
    33d0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    33d4:	d65f03c0 	ret
    33d8:	d503201f 	nop
    33dc:	d503201f 	nop

00000000000033e0 <sptSparseTensorDensity>:
#include <bits/types/FILE.h>
#include <stdio.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
    33e0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    33e4:	910003fd 	mov	x29, sp
    33e8:	a90153f3 	stp	x19, x20, [sp, #16]
    33ec:	aa0003f4 	mov	x20, x0
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    33f0:	b9400013 	ldr	w19, [x0]
    33f4:	fd400c00 	ldr	d0, [x0, #24]
{
    33f8:	fd0013e8 	str	d8, [sp, #32]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    33fc:	1e630261 	ucvtf	d1, w19
    3400:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
    3404:	7e61d800 	ucvtf	d0, d0
    3408:	1e611901 	fdiv	d1, d8, d1
    340c:	97fff789 	bl	1230 <pow@plt>
	double density = 1.0;
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3410:	34000793 	cbz	w19, 3500 <sptSparseTensorDensity+0x120>
    3414:	51000660 	sub	w0, w19, #0x1
    3418:	1e604005 	fmov	d5, d0
		density *= root / (double)tsr->ndims[m];
    341c:	f9400a82 	ldr	x2, [x20, #16]
    3420:	71000c1f 	cmp	w0, #0x3
    3424:	54000789 	b.ls	3514 <sptSparseTensorDensity+0x134>  // b.plast
    3428:	53027e61 	lsr	w1, w19, #2
    342c:	aa0203e0 	mov	x0, x2
    3430:	4e080404 	dup	v4.2d, v0.d[0]
    3434:	6f03f603 	fmov	v3.2d, #1.000000000000000000e+00
    3438:	8b215041 	add	x1, x2, w1, uxtw #4
    343c:	d503201f 	nop
    3440:	3cc10401 	ldr	q1, [x0], #16
    3444:	2f20a422 	uxtl	v2.2d, v1.2s
    3448:	6f20a421 	uxtl2	v1.2d, v1.4s
    344c:	4e61d842 	scvtf	v2.2d, v2.2d
    3450:	4e61d821 	scvtf	v1.2d, v1.2d
    3454:	6e62fc82 	fdiv	v2.2d, v4.2d, v2.2d
    3458:	6e61fc81 	fdiv	v1.2d, v4.2d, v1.2d
    345c:	6e61dc41 	fmul	v1.2d, v2.2d, v1.2d
    3460:	6e61dc63 	fmul	v3.2d, v3.2d, v1.2d
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3464:	eb01001f 	cmp	x0, x1
    3468:	54fffec1 	b.ne	3440 <sptSparseTensorDensity+0x60>  // b.any
    346c:	6f00e401 	movi	v1.2d, #0x0
    3470:	121e7660 	and	w0, w19, #0xfffffffc
    3474:	6e014061 	ext	v1.16b, v3.16b, v1.16b, #8
    3478:	6e63dc21 	fmul	v1.2d, v1.2d, v3.2d
    347c:	5e080420 	mov	d0, v1.d[0]
    3480:	f240067f 	tst	x19, #0x3
    3484:	54000360 	b.eq	34f0 <sptSparseTensorDensity+0x110>  // b.none
		density *= root / (double)tsr->ndims[m];
    3488:	2a0003e1 	mov	w1, w0
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    348c:	11000403 	add	w3, w0, #0x1
		density *= root / (double)tsr->ndims[m];
    3490:	b8617841 	ldr	w1, [x2, x1, lsl #2]
    3494:	1e630021 	ucvtf	d1, w1
    3498:	1e6118a1 	fdiv	d1, d5, d1
    349c:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34a0:	6b03027f 	cmp	w19, w3
    34a4:	54000269 	b.ls	34f0 <sptSparseTensorDensity+0x110>  // b.plast
		density *= root / (double)tsr->ndims[m];
    34a8:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34ac:	11000801 	add	w1, w0, #0x2
		density *= root / (double)tsr->ndims[m];
    34b0:	1e630061 	ucvtf	d1, w3
    34b4:	1e6118a1 	fdiv	d1, d5, d1
    34b8:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34bc:	6b01027f 	cmp	w19, w1
    34c0:	54000189 	b.ls	34f0 <sptSparseTensorDensity+0x110>  // b.plast
		density *= root / (double)tsr->ndims[m];
    34c4:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34c8:	11000c00 	add	w0, w0, #0x3
		density *= root / (double)tsr->ndims[m];
    34cc:	1e630021 	ucvtf	d1, w1
    34d0:	1e6118a1 	fdiv	d1, d5, d1
    34d4:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34d8:	6b00027f 	cmp	w19, w0
    34dc:	540000a9 	b.ls	34f0 <sptSparseTensorDensity+0x110>  // b.plast
		density *= root / (double)tsr->ndims[m];
    34e0:	b8605840 	ldr	w0, [x2, w0, uxtw #2]
    34e4:	1e630001 	ucvtf	d1, w0
    34e8:	1e6118a5 	fdiv	d5, d5, d1
    34ec:	1e650800 	fmul	d0, d0, d5
	}

	return density;
}
    34f0:	a94153f3 	ldp	x19, x20, [sp, #16]
    34f4:	fd4013e8 	ldr	d8, [sp, #32]
    34f8:	a8c37bfd 	ldp	x29, x30, [sp], #48
    34fc:	d65f03c0 	ret
	double density = 1.0;
    3500:	1e604100 	fmov	d0, d8
}
    3504:	fd4013e8 	ldr	d8, [sp, #32]
    3508:	a94153f3 	ldp	x19, x20, [sp, #16]
    350c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    3510:	d65f03c0 	ret
	double density = 1.0;
    3514:	1e604100 	fmov	d0, d8
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3518:	52800000 	mov	w0, #0x0                   	// #0
    351c:	17ffffdb 	b	3488 <sptSparseTensorDensity+0xa8>

0000000000003520 <sptSparseTensorStatus>:



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
    3520:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3524:	aa0103e3 	mov	x3, x1
    3528:	d2800822 	mov	x2, #0x41                  	// #65
{
    352c:	910003fd 	mov	x29, sp
    3530:	a90153f3 	stp	x19, x20, [sp, #16]
    3534:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3538:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    353c:	91178000 	add	x0, x0, #0x5e0
{
    3540:	a9025bf5 	stp	x21, x22, [sp, #32]
    3544:	aa0103f5 	mov	x21, x1
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3548:	d2800021 	mov	x1, #0x1                   	// #1
{
    354c:	fd001be8 	str	d8, [sp, #48]
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3550:	97fff7a0 	bl	13d0 <fwrite@plt>
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3554:	d0000016 	adrp	x22, 5000 <_IO_stdin_used+0x120>
    3558:	f9400a62 	ldr	x2, [x19, #16]
    355c:	aa1503e0 	mov	x0, x21
    3560:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    3564:	9118a021 	add	x1, x1, #0x628
    3568:	911922d6 	add	x22, x22, #0x648
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    356c:	52800034 	mov	w20, #0x1                   	// #1
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3570:	b9400042 	ldr	w2, [x2]
    3574:	97fff7bb 	bl	1460 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3578:	b9400260 	ldr	w0, [x19]
    357c:	6b14001f 	cmp	w0, w20
    3580:	54000169 	b.ls	35ac <sptSparseTensorStatus+0x8c>  // b.plast
    3584:	d503201f 	nop
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    3588:	f9400a62 	ldr	x2, [x19, #16]
    358c:	aa1603e1 	mov	x1, x22
    3590:	aa1503e0 	mov	x0, x21
    3594:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3598:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    359c:	97fff7b1 	bl	1460 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    35a0:	b9400260 	ldr	w0, [x19]
    35a4:	6b14001f 	cmp	w0, w20
    35a8:	54ffff08 	b.hi	3588 <sptSparseTensorStatus+0x68>  // b.pmore
	}
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    35ac:	f9400e62 	ldr	x2, [x19, #24]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    35b0:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    35b4:	aa1503e0 	mov	x0, x21
    35b8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    35bc:	9118e021 	add	x1, x1, #0x638
    35c0:	97fff7a8 	bl	1460 <fprintf@plt>
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    35c4:	b9400274 	ldr	w20, [x19]
    35c8:	fd400e60 	ldr	d0, [x19, #24]
    35cc:	1e630281 	ucvtf	d1, w20
    35d0:	7e61d800 	ucvtf	d0, d0
    35d4:	1e611901 	fdiv	d1, d8, d1
    35d8:	97fff716 	bl	1230 <pow@plt>
    35dc:	1e604005 	fmov	d5, d0
	double density = 1.0;
    35e0:	1e604100 	fmov	d0, d8
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    35e4:	340006f4 	cbz	w20, 36c0 <sptSparseTensorStatus+0x1a0>
    35e8:	51000680 	sub	w0, w20, #0x1
		density *= root / (double)tsr->ndims[m];
    35ec:	f9400a62 	ldr	x2, [x19, #16]
    35f0:	71000c1f 	cmp	w0, #0x3
    35f4:	54000e49 	b.ls	37bc <sptSparseTensorStatus+0x29c>  // b.plast
    35f8:	53027e81 	lsr	w1, w20, #2
    35fc:	aa0203e0 	mov	x0, x2
    3600:	4e0804a4 	dup	v4.2d, v5.d[0]
    3604:	6f03f603 	fmov	v3.2d, #1.000000000000000000e+00
    3608:	8b215041 	add	x1, x2, w1, uxtw #4
    360c:	d503201f 	nop
    3610:	3cc10401 	ldr	q1, [x0], #16
    3614:	6f20a422 	uxtl2	v2.2d, v1.4s
    3618:	2f20a421 	uxtl	v1.2d, v1.2s
    361c:	4e61d842 	scvtf	v2.2d, v2.2d
    3620:	4e61d821 	scvtf	v1.2d, v1.2d
    3624:	6e62fc82 	fdiv	v2.2d, v4.2d, v2.2d
    3628:	6e61fc81 	fdiv	v1.2d, v4.2d, v1.2d
    362c:	6e61dc41 	fmul	v1.2d, v2.2d, v1.2d
    3630:	6e61dc63 	fmul	v3.2d, v3.2d, v1.2d
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3634:	eb01001f 	cmp	x0, x1
    3638:	54fffec1 	b.ne	3610 <sptSparseTensorStatus+0xf0>  // b.any
    363c:	6f00e401 	movi	v1.2d, #0x0
    3640:	121e7680 	and	w0, w20, #0xfffffffc
    3644:	6e014061 	ext	v1.16b, v3.16b, v1.16b, #8
    3648:	6e63dc21 	fmul	v1.2d, v1.2d, v3.2d
    364c:	5e080420 	mov	d0, v1.d[0]
    3650:	f240069f 	tst	x20, #0x3
    3654:	54000360 	b.eq	36c0 <sptSparseTensorStatus+0x1a0>  // b.none
		density *= root / (double)tsr->ndims[m];
    3658:	2a0003e1 	mov	w1, w0
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    365c:	11000403 	add	w3, w0, #0x1
		density *= root / (double)tsr->ndims[m];
    3660:	b8617841 	ldr	w1, [x2, x1, lsl #2]
    3664:	1e630021 	ucvtf	d1, w1
    3668:	1e6118a1 	fdiv	d1, d5, d1
    366c:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3670:	6b03029f 	cmp	w20, w3
    3674:	54000269 	b.ls	36c0 <sptSparseTensorStatus+0x1a0>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3678:	b8635843 	ldr	w3, [x2, w3, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    367c:	11000801 	add	w1, w0, #0x2
		density *= root / (double)tsr->ndims[m];
    3680:	1e630061 	ucvtf	d1, w3
    3684:	1e6118a1 	fdiv	d1, d5, d1
    3688:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    368c:	6b01029f 	cmp	w20, w1
    3690:	54000189 	b.ls	36c0 <sptSparseTensorStatus+0x1a0>  // b.plast
		density *= root / (double)tsr->ndims[m];
    3694:	b8617841 	ldr	w1, [x2, x1, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    3698:	11000c00 	add	w0, w0, #0x3
		density *= root / (double)tsr->ndims[m];
    369c:	1e630021 	ucvtf	d1, w1
    36a0:	1e6118a1 	fdiv	d1, d5, d1
    36a4:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    36a8:	6b14001f 	cmp	w0, w20
    36ac:	540000a2 	b.cs	36c0 <sptSparseTensorStatus+0x1a0>  // b.hs, b.nlast
		density *= root / (double)tsr->ndims[m];
    36b0:	b8605840 	ldr	w0, [x2, w0, uxtw #2]
    36b4:	1e630001 	ucvtf	d1, w0
    36b8:	1e6118a5 	fdiv	d5, d5, d1
    36bc:	1e650800 	fmul	d0, d0, d5
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));
    36c0:	aa1503e0 	mov	x0, x21
    36c4:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    36c8:	91194021 	add	x1, x1, #0x650
    36cc:	97fff765 	bl	1460 <fprintf@plt>

	fprintf(fp, "Average slice length (c): ");
    36d0:	aa1503e3 	mov	x3, x21
    36d4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    36d8:	d2800342 	mov	x2, #0x1a                  	// #26
    36dc:	91198000 	add	x0, x0, #0x660
    36e0:	d2800021 	mov	x1, #0x1                   	// #1
    36e4:	97fff73b 	bl	13d0 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    36e8:	b9400260 	ldr	w0, [x19]
    36ec:	7100041f 	cmp	w0, #0x1
    36f0:	54000620 	b.eq	37b4 <sptSparseTensorStatus+0x294>  // b.none
    36f4:	d0000016 	adrp	x22, 5000 <_IO_stdin_used+0x120>
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    36f8:	911a02d6 	add	x22, x22, #0x680
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    36fc:	52800014 	mov	w20, #0x0                   	// #0
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3700:	f9400a62 	ldr	x2, [x19, #16]
    3704:	aa1603e1 	mov	x1, x22
    3708:	fd400e60 	ldr	d0, [x19, #24]
    370c:	aa1503e0 	mov	x0, x21
    3710:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3714:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3718:	7e61d800 	ucvtf	d0, d0
    371c:	1e630041 	ucvtf	d1, w2
    3720:	1e611800 	fdiv	d0, d0, d1
    3724:	97fff74f 	bl	1460 <fprintf@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3728:	b9400262 	ldr	w2, [x19]
    372c:	51000442 	sub	w2, w2, #0x1
    3730:	6b14005f 	cmp	w2, w20
    3734:	54fffe68 	b.hi	3700 <sptSparseTensorStatus+0x1e0>  // b.pmore
	}
	fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);
    3738:	d37e7c42 	ubfiz	x2, x2, #2, #32
    373c:	f9400a60 	ldr	x0, [x19, #16]
    3740:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    3744:	fd400e60 	ldr	d0, [x19, #24]
    3748:	911a4021 	add	x1, x1, #0x690
    374c:	b8626802 	ldr	w2, [x0, x2]
    3750:	aa1503e0 	mov	x0, x21
    3754:	7e61d800 	ucvtf	d0, d0
    3758:	1e630041 	ucvtf	d1, w2
    375c:	1e611800 	fdiv	d0, d0, d1
    3760:	97fff740 	bl	1460 <fprintf@plt>

	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    3764:	b9400261 	ldr	w1, [x19]
    3768:	f9400e60 	ldr	x0, [x19, #24]
    376c:	9b000020 	madd	x0, x1, x0, x0
    3770:	d37ef400 	lsl	x0, x0, #2
    3774:	94000543 	bl	4c80 <sptBytesString>
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3778:	aa0003e2 	mov	x2, x0
	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    377c:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3780:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    3784:	aa1503e0 	mov	x0, x21
    3788:	911a6021 	add	x1, x1, #0x698
    378c:	97fff735 	bl	1460 <fprintf@plt>
	fprintf(fp, "\n");
    3790:	aa1503e1 	mov	x1, x21
    3794:	52800140 	mov	w0, #0xa                   	// #10
    3798:	97fff69a 	bl	1200 <fputc@plt>
	free(bytestr);
    379c:	aa1303e0 	mov	x0, x19
}
    37a0:	fd401be8 	ldr	d8, [sp, #48]
    37a4:	a94153f3 	ldp	x19, x20, [sp, #16]
    37a8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    37ac:	a8c47bfd 	ldp	x29, x30, [sp], #64
	free(bytestr);
    37b0:	17fff704 	b	13c0 <free@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    37b4:	d2800002 	mov	x2, #0x0                   	// #0
    37b8:	17ffffe1 	b	373c <sptSparseTensorStatus+0x21c>
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    37bc:	52800000 	mov	w0, #0x0                   	// #0
    37c0:	17ffffa6 	b	3658 <sptSparseTensorStatus+0x138>
    37c4:	d503201f 	nop
    37c8:	d503201f 	nop
    37cc:	d503201f 	nop

00000000000037d0 <sptMTTKRP_3D>:

int sptMTTKRP_3D(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode)
{
    37d0:	a9b87bfd 	stp	x29, x30, [sp, #-128]!
    37d4:	910003fd 	mov	x29, sp
    37d8:	a9046bf9 	stp	x25, x26, [sp, #64]
    37dc:	aa0003fa 	mov	x26, x0
    37e0:	aa0103f9 	mov	x25, x1
	sptIndex const nmodes = X->nmodes;
	sptNnzIndex const nnz = X->nnz;
	sptIndex const * const ndims = X->ndims;
	sptValue const * const restrict vals = X->values.data;
	sptIndex const stride = mats[0]->stride;
    37e4:	f9400020 	ldr	x0, [x1]
{
    37e8:	a90153f3 	stp	x19, x20, [sp, #16]

	/* Check the mats. */
	sptAssert(nmodes ==3);
    37ec:	b9400341 	ldr	w1, [x26]
{
    37f0:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const stride = mats[0]->stride;
    37f4:	b9400c14 	ldr	w20, [x0, #12]
{
    37f8:	a90363f7 	stp	x23, x24, [sp, #48]
    37fc:	a90573fb 	stp	x27, x28, [sp, #80]
    3800:	aa0203fc 	mov	x28, x2
	sptNnzIndex const nnz = X->nnz;
    3804:	a9415742 	ldp	x2, x21, [x26, #16]
	sptValue const * const restrict vals = X->values.data;
    3808:	f9401f57 	ldr	x23, [x26, #56]
	sptAssert(nmodes ==3);
    380c:	71000c3f 	cmp	w1, #0x3
    3810:	54001061 	b.ne	3a1c <sptMTTKRP_3D+0x24c>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3814:	f9400f24 	ldr	x4, [x25, #24]
    3818:	b9400405 	ldr	w5, [x0, #4]
    381c:	b9400481 	ldr	w1, [x4, #4]
    3820:	6b05003f 	cmp	w1, w5
    3824:	54000d01 	b.ne	39c4 <sptMTTKRP_3D+0x1f4>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
		}
		if(mats[i]->nrows != ndims[i]) {
    3828:	b9400000 	ldr	w0, [x0]
    382c:	b9400045 	ldr	w5, [x2]
    3830:	6b0000bf 	cmp	w5, w0
    3834:	54000de1 	b.ne	39f0 <sptMTTKRP_3D+0x220>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3838:	f9400720 	ldr	x0, [x25, #8]
    383c:	b9400405 	ldr	w5, [x0, #4]
    3840:	6b05003f 	cmp	w1, w5
    3844:	54000c01 	b.ne	39c4 <sptMTTKRP_3D+0x1f4>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3848:	b9400000 	ldr	w0, [x0]
    384c:	b9400445 	ldr	w5, [x2, #4]
    3850:	6b0000bf 	cmp	w5, w0
    3854:	54000ce1 	b.ne	39f0 <sptMTTKRP_3D+0x220>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3858:	f9400b20 	ldr	x0, [x25, #16]
    385c:	b9400405 	ldr	w5, [x0, #4]
    3860:	6b0100bf 	cmp	w5, w1
    3864:	54000b01 	b.ne	39c4 <sptMTTKRP_3D+0x1f4>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3868:	b9400001 	ldr	w1, [x0]
    386c:	b9400840 	ldr	w0, [x2, #8]
    3870:	6b00003f 	cmp	w1, w0
    3874:	54000be1 	b.ne	39f0 <sptMTTKRP_3D+0x220>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}


	sptIndex const tmpI = mats[mode]->nrows;
    3878:	f8635b20 	ldr	x0, [x25, w3, uxtw #3]
	sptIndex const R = mats[mode]->ncols;
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    387c:	5280031b 	mov	w27, #0x18                  	// #24
    3880:	f9401345 	ldr	x5, [x26, #32]
	sptMatrix * const restrict M = mats[nmodes];
	sptValue * const restrict mvals = M->values;
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3884:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    3888:	29405802 	ldp	w2, w22, [x0]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    388c:	9bbb1463 	umaddl	x3, w3, w27, x5
	sptValue * const restrict mvals = M->values;
    3890:	f9400893 	ldr	x19, [x4, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3894:	1b027e82 	mul	w2, w20, w2
    3898:	aa1303e0 	mov	x0, x19
    389c:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    38a0:	f9400878 	ldr	x24, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    38a4:	fd0033e8 	str	d8, [sp, #96]
    38a8:	97fff68a 	bl	12d0 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
	sptIndex times_mat_index_2 = mats_order[2];
    38ac:	29408f82 	ldp	w2, w3, [x28, #4]
	sptIndex mode_i;
	sptIndex tmp_i_1, tmp_i_2;
	sptValue entry;

	sptTimer timer;
	sptNewTimer(&timer, 0);
    38b0:	9101e3e0 	add	x0, sp, #0x78
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    38b4:	f9401344 	ldr	x4, [x26, #32]
	sptNewTimer(&timer, 0);
    38b8:	52800001 	mov	w1, #0x0                   	// #0
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    38bc:	f8625b3c 	ldr	x28, [x25, w2, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    38c0:	9bbb1042 	umaddl	x2, w2, w27, x4
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    38c4:	9bbb107b 	umaddl	x27, w3, w27, x4
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    38c8:	f8635b3a 	ldr	x26, [x25, w3, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    38cc:	f9400859 	ldr	x25, [x2, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    38d0:	f9400b7b 	ldr	x27, [x27, #16]
	sptNewTimer(&timer, 0);
    38d4:	94000363 	bl	4660 <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    38d8:	f9403fe0 	ldr	x0, [sp, #120]
    38dc:	94000381 	bl	46e0 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    38e0:	b4000435 	cbz	x21, 3964 <sptMTTKRP_3D+0x194>
    38e4:	34000416 	cbz	w22, 3964 <sptMTTKRP_3D+0x194>
		tmp_i_1 = times_inds_1[x];
		tmp_i_2 = times_inds_2[x];
		entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    38e8:	f9400b47 	ldr	x7, [x26, #16]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    38ec:	d2800009 	mov	x9, #0x0                   	// #0
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    38f0:	f9400b88 	ldr	x8, [x28, #16]
    38f4:	d503201f 	nop
    38f8:	b8697b04 	ldr	w4, [x24, x9, lsl #2]
    38fc:	b8697b26 	ldr	w6, [x25, x9, lsl #2]
    3900:	b8697b65 	ldr	w5, [x27, x9, lsl #2]
		entry = vals[x];
    3904:	bc697ae3 	ldr	s3, [x23, x9, lsl #2]
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3908:	1b047e84 	mul	w4, w20, w4
    390c:	1b067e86 	mul	w6, w20, w6
    3910:	1b057e85 	mul	w5, w20, w5
    3914:	2a0403e0 	mov	w0, w4
    3918:	4b0000c6 	sub	w6, w6, w0
    391c:	0b160084 	add	w4, w4, w22
    3920:	4b0000a5 	sub	w5, w5, w0
    3924:	d503201f 	nop
    3928:	0b0000c3 	add	w3, w6, w0
    392c:	0b0000a2 	add	w2, w5, w0
    3930:	2a0003e1 	mov	w1, w0
		for(sptIndex r=0; r<R; ++r) {
    3934:	11000400 	add	w0, w0, #0x1
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3938:	bc637901 	ldr	s1, [x8, x3, lsl #2]
    393c:	bc6278e2 	ldr	s2, [x7, x2, lsl #2]
    3940:	bc617a60 	ldr	s0, [x19, x1, lsl #2]
    3944:	1e220821 	fmul	s1, s1, s2
    3948:	1f030020 	fmadd	s0, s1, s3, s0
    394c:	bc217a60 	str	s0, [x19, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3950:	6b04001f 	cmp	w0, w4
    3954:	54fffea1 	b.ne	3928 <sptMTTKRP_3D+0x158>  // b.any
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3958:	91000529 	add	x9, x9, #0x1
    395c:	eb0902bf 	cmp	x21, x9
    3960:	54fffcc1 	b.ne	38f8 <sptMTTKRP_3D+0x128>  // b.any
		}
	}
	sptStopTimer(timer);
    3964:	f9403fe0 	ldr	x0, [sp, #120]
    3968:	9400038e 	bl	47a0 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    396c:	f9403fe0 	ldr	x0, [sp, #120]
    3970:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    3974:	911c0021 	add	x1, x1, #0x700
    3978:	940003ca 	bl	48a0 <sptPrintElapsedTime>
    397c:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3980:	f9403fe0 	ldr	x0, [sp, #120]
    3984:	94000403 	bl	4990 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3988:	1e604100 	fmov	d0, d8
    398c:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3990:	911ce000 	add	x0, x0, #0x738
    3994:	97fff6a3 	bl	1420 <printf@plt>
	printf("\n");
    3998:	52800140 	mov	w0, #0xa                   	// #10
    399c:	97fff6ad 	bl	1450 <putchar@plt>

	return 0;
    39a0:	fd4033e8 	ldr	d8, [sp, #96]
    39a4:	52800000 	mov	w0, #0x0                   	// #0
}
    39a8:	a94153f3 	ldp	x19, x20, [sp, #16]
    39ac:	a9425bf5 	ldp	x21, x22, [sp, #32]
    39b0:	a94363f7 	ldp	x23, x24, [sp, #48]
    39b4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    39b8:	a94573fb 	ldp	x27, x28, [sp, #80]
    39bc:	a8c87bfd 	ldp	x29, x30, [sp], #128
    39c0:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    39c4:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    39c8:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    39cc:	911ac084 	add	x4, x4, #0x6b0
    39d0:	911b6042 	add	x2, x2, #0x6d8
    39d4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    39d8:	52801263 	mov	w3, #0x93                  	// #147
    39dc:	911c0000 	add	x0, x0, #0x700
    39e0:	52800041 	mov	w1, #0x2                   	// #2
    39e4:	9400046b 	bl	4b90 <spt_ComplainError>
    39e8:	52800040 	mov	w0, #0x2                   	// #2
    39ec:	17ffffef 	b	39a8 <sptMTTKRP_3D+0x1d8>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    39f0:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    39f4:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    39f8:	911c6084 	add	x4, x4, #0x718
    39fc:	911b6042 	add	x2, x2, #0x6d8
    3a00:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3a04:	528012c3 	mov	w3, #0x96                  	// #150
    3a08:	911c0000 	add	x0, x0, #0x700
    3a0c:	52800041 	mov	w1, #0x2                   	// #2
    3a10:	94000460 	bl	4b90 <spt_ComplainError>
    3a14:	52800040 	mov	w0, #0x2                   	// #2
    3a18:	17ffffe4 	b	39a8 <sptMTTKRP_3D+0x1d8>
	sptAssert(nmodes ==3);
    3a1c:	12800000 	mov	w0, #0xffffffff            	// #-1
    3a20:	fd0033e8 	str	d8, [sp, #96]
    3a24:	97fff5ef 	bl	11e0 <exit@plt>
    3a28:	d503201f 	nop
    3a2c:	d503201f 	nop

0000000000003a30 <sptMTTKRP>:
							sptIndex const mode) {
    3a30:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    3a34:	910003fd 	mov	x29, sp
    3a38:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const nmodes = X->nmodes;
    3a3c:	b9400015 	ldr	w21, [x0]
	if(nmodes == 3) {
    3a40:	71000ebf 	cmp	w21, #0x3
    3a44:	54002060 	b.eq	3e50 <sptMTTKRP+0x420>  // b.none
	sptNnzIndex const nnz = X->nnz;
    3a48:	a90363f7 	stp	x23, x24, [sp, #48]
    3a4c:	aa0103f6 	mov	x22, x1
    3a50:	aa0203f8 	mov	x24, x2
    3a54:	a90573fb 	stp	x27, x28, [sp, #80]
    3a58:	2a0303e5 	mov	w5, w3
    3a5c:	aa0003fc 	mov	x28, x0
	sptMatrix * const restrict M = mats[nmodes];
    3a60:	f8755826 	ldr	x6, [x1, w21, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    3a64:	f9400804 	ldr	x4, [x0, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    3a68:	340003d5 	cbz	w21, 3ae0 <sptMTTKRP+0xb0>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3a6c:	b94004c3 	ldr	w3, [x6, #4]
    3a70:	d2800000 	mov	x0, #0x0                   	// #0
    3a74:	14000008 	b	3a94 <sptMTTKRP+0x64>
		if(mats[i]->nrows != ndims[i]) {
    3a78:	b9400022 	ldr	w2, [x1]
    3a7c:	b8607881 	ldr	w1, [x4, x0, lsl #2]
    3a80:	6b01005f 	cmp	w2, w1
    3a84:	54001d81 	b.ne	3e34 <sptMTTKRP+0x404>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3a88:	91000400 	add	x0, x0, #0x1
    3a8c:	6b0002bf 	cmp	w21, w0
    3a90:	54000289 	b.ls	3ae0 <sptMTTKRP+0xb0>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3a94:	f8607ac1 	ldr	x1, [x22, x0, lsl #3]
    3a98:	b9400422 	ldr	w2, [x1, #4]
    3a9c:	6b03005f 	cmp	w2, w3
    3aa0:	54fffec0 	b.eq	3a78 <sptMTTKRP+0x48>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3aa4:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    3aa8:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    3aac:	911ac084 	add	x4, x4, #0x6b0
    3ab0:	911b6042 	add	x2, x2, #0x6d8
    3ab4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3ab8:	528007c3 	mov	w3, #0x3e                  	// #62
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3abc:	911c0000 	add	x0, x0, #0x700
    3ac0:	52800041 	mov	w1, #0x2                   	// #2
    3ac4:	94000433 	bl	4b90 <spt_ComplainError>
    3ac8:	a94363f7 	ldp	x23, x24, [sp, #48]
    3acc:	52800040 	mov	w0, #0x2                   	// #2
    3ad0:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3ad4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3ad8:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3adc:	d65f03c0 	ret
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3ae0:	f9401383 	ldr	x3, [x28, #32]
    3ae4:	a9046bf9 	stp	x25, x26, [sp, #64]
    3ae8:	52800319 	mov	w25, #0x18                  	// #24
	sptIndex const tmpI = mats[mode]->nrows;
    3aec:	f8655ac0 	ldr	x0, [x22, w5, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3af0:	9bb90ca3 	umaddl	x3, w5, w25, x3
	sptIndex const stride = mats[0]->stride;
    3af4:	f94002c4 	ldr	x4, [x22]
	sptNewValueVector(&scratch, R, R);
    3af8:	910223e2 	add	x2, sp, #0x88
    3afc:	a90153f3 	stp	x19, x20, [sp, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3b00:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    3b04:	b9400413 	ldr	w19, [x0, #4]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3b08:	f940087b 	ldr	x27, [x3, #16]
	sptNnzIndex const nnz = X->nnz;
    3b0c:	f9400f83 	ldr	x3, [x28, #24]
	sptNewValueVector(&scratch, R, R);
    3b10:	a9070be3 	stp	x3, x2, [sp, #112]
	sptIndex const stride = mats[0]->stride;
    3b14:	b9400c97 	ldr	w23, [x4, #12]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3b18:	b9400002 	ldr	w2, [x0]
	sptValue * const restrict mvals = M->values;
    3b1c:	f94008d4 	ldr	x20, [x6, #16]
	sptValue const * const restrict vals = X->values.data;
    3b20:	f9401f9a 	ldr	x26, [x28, #56]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3b24:	1b027ee2 	mul	w2, w23, w2
    3b28:	aa1403e0 	mov	x0, x20
    3b2c:	fd0033e8 	str	d8, [sp, #96]
    3b30:	d37ef442 	lsl	x2, x2, #2
    3b34:	97fff5e7 	bl	12d0 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    3b38:	f9403fe0 	ldr	x0, [sp, #120]
    3b3c:	2a1303e2 	mov	w2, w19
    3b40:	aa0203e1 	mov	x1, x2
    3b44:	97fff94b 	bl	2070 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    3b48:	0f000400 	movi	v0.2s, #0x0
    3b4c:	f9403fe0 	ldr	x0, [sp, #120]
    3b50:	97fff970 	bl	2110 <sptConstantValueVector>
	sptNewTimer(&timer, 0);
    3b54:	52800001 	mov	w1, #0x0                   	// #0
    3b58:	910203e0 	add	x0, sp, #0x80
    3b5c:	940002c1 	bl	4660 <sptNewTimer>
	sptStartTimer(timer);
    3b60:	f94043e0 	ldr	x0, [sp, #128]
    3b64:	940002df 	bl	46e0 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3b68:	f9403be3 	ldr	x3, [sp, #112]
    3b6c:	b40010a3 	cbz	x3, 3d80 <sptMTTKRP+0x350>
		sptIndex times_mat_index = mats_order[1];
    3b70:	b9400700 	ldr	w0, [x24, #4]
    3b74:	53027e64 	lsr	w4, w19, #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3b78:	f940138b 	ldr	x11, [x28, #32]
    3b7c:	121e766d 	and	w13, w19, #0xfffffffc
			scratch.data[r] = entry * times_mat_values[r];
    3b80:	f9404fe1 	ldr	x1, [sp, #152]
    3b84:	51000484 	sub	w4, w4, #0x1
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3b88:	9bb92c02 	umaddl	x2, w0, w25, x11
		for(sptIndex r=0; r<R; ++r) {
    3b8c:	110009b2 	add	w18, w13, #0x2
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3b90:	f8605ac0 	ldr	x0, [x22, w0, uxtw #3]
    3b94:	91000484 	add	x4, x4, #0x1
			scratch.data[r] = entry * times_mat_values[r];
    3b98:	110005b0 	add	w16, w13, #0x1
    3b9c:	927e766e 	and	x14, x19, #0xfffffffc
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3ba0:	f940085c 	ldr	x28, [x2, #16]
		for(sptIndex r=0; r<R; ++r) {
    3ba4:	aa1203f1 	mov	x17, x18
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3ba8:	f940081e 	ldr	x30, [x0, #16]
    3bac:	d37cec84 	lsl	x4, x4, #4
		for(sptIndex r=0; r<R; ++r) {
    3bb0:	110005af 	add	w15, w13, #0x1
    3bb4:	5100066a 	sub	w10, w19, #0x1
			scratch.data[r] = entry * times_mat_values[r];
    3bb8:	8b12082c 	add	x12, x1, x18, lsl #2
    3bbc:	8b100829 	add	x9, x1, x16, lsl #2
    3bc0:	8b0e0828 	add	x8, x1, x14, lsl #2
    3bc4:	d37ef463 	lsl	x3, x3, #2
    3bc8:	d2800007 	mov	x7, #0x0                   	// #0
    3bcc:	f9003be3 	str	x3, [sp, #112]
		sptIndex tmp_mult = tmp_i * stride;
    3bd0:	b8676b83 	ldr	w3, [x28, x7]
		sptValue const entry = vals[x];
    3bd4:	bc676b42 	ldr	s2, [x26, x7]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3bd8:	1b037ee3 	mul	w3, w23, w3
		for(sptIndex r=0; r<R; ++r) {
    3bdc:	34000cb3 	cbz	w19, 3d70 <sptMTTKRP+0x340>
    3be0:	91000460 	add	x0, x3, #0x1
    3be4:	d37ef400 	lsl	x0, x0, #2
    3be8:	8b0003c2 	add	x2, x30, x0
    3bec:	cb020022 	sub	x2, x1, x2
    3bf0:	f100205f 	cmp	x2, #0x8
    3bf4:	7a458940 	ccmp	w10, #0x5, #0x0, hi  // hi = pmore
    3bf8:	540010c9 	b.ls	3e10 <sptMTTKRP+0x3e0>  // b.plast
    3bfc:	4e040441 	dup	v1.4s, v2.s[0]
    3c00:	d1001002 	sub	x2, x0, #0x4
    3c04:	8b0203c2 	add	x2, x30, x2
    3c08:	d2800000 	mov	x0, #0x0                   	// #0
    3c0c:	d503201f 	nop
			scratch.data[r] = entry * times_mat_values[r];
    3c10:	3ce06840 	ldr	q0, [x2, x0]
    3c14:	6e21dc00 	fmul	v0.4s, v0.4s, v1.4s
    3c18:	3ca06820 	str	q0, [x1, x0]
		for(sptIndex r=0; r<R; ++r) {
    3c1c:	91004000 	add	x0, x0, #0x10
    3c20:	eb04001f 	cmp	x0, x4
    3c24:	54ffff61 	b.ne	3c10 <sptMTTKRP+0x1e0>  // b.any
    3c28:	6b1301bf 	cmp	w13, w19
    3c2c:	54000220 	b.eq	3c70 <sptMTTKRP+0x240>  // b.none
			scratch.data[r] = entry * times_mat_values[r];
    3c30:	8b0e0060 	add	x0, x3, x14
    3c34:	bc607bc0 	ldr	s0, [x30, x0, lsl #2]
    3c38:	1e220800 	fmul	s0, s0, s2
    3c3c:	bd000100 	str	s0, [x8]
		for(sptIndex r=0; r<R; ++r) {
    3c40:	6b0f027f 	cmp	w19, w15
    3c44:	54000169 	b.ls	3c70 <sptMTTKRP+0x240>  // b.plast
			scratch.data[r] = entry * times_mat_values[r];
    3c48:	8b100060 	add	x0, x3, x16
    3c4c:	bc607bc0 	ldr	s0, [x30, x0, lsl #2]
    3c50:	1e220800 	fmul	s0, s0, s2
    3c54:	bd000120 	str	s0, [x9]
		for(sptIndex r=0; r<R; ++r) {
    3c58:	6b11027f 	cmp	w19, w17
    3c5c:	540000a9 	b.ls	3c70 <sptMTTKRP+0x240>  // b.plast
			scratch.data[r] = entry * times_mat_values[r];
    3c60:	8b120063 	add	x3, x3, x18
    3c64:	bc637bc0 	ldr	s0, [x30, x3, lsl #2]
    3c68:	1e220802 	fmul	s2, s0, s2
    3c6c:	bd000182 	str	s2, [x12]
		for(sptIndex i=2; i<nmodes; ++i) {
    3c70:	d2800046 	mov	x6, #0x2                   	// #2
    3c74:	71000abf 	cmp	w21, #0x2
    3c78:	54000669 	b.ls	3d44 <sptMTTKRP+0x314>  // b.plast
    3c7c:	d503201f 	nop
			times_mat_index = mats_order[i];
    3c80:	b8667b00 	ldr	w0, [x24, x6, lsl #2]
			tmp_i = times_inds[x];
    3c84:	9bb92c03 	umaddl	x3, w0, w25, x11
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3c88:	f8605ac2 	ldr	x2, [x22, w0, uxtw #3]
			tmp_i = times_inds[x];
    3c8c:	f9400860 	ldr	x0, [x3, #16]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3c90:	f9400845 	ldr	x5, [x2, #16]
			tmp_mult = tmp_i * stride;
    3c94:	b8676803 	ldr	w3, [x0, x7]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3c98:	1b037ee3 	mul	w3, w23, w3
			for(sptIndex r=0; r<R; ++r) {
    3c9c:	91000462 	add	x2, x3, #0x1
    3ca0:	d37ef442 	lsl	x2, x2, #2
    3ca4:	8b0200a0 	add	x0, x5, x2
    3ca8:	cb000020 	sub	x0, x1, x0
    3cac:	f100201f 	cmp	x0, #0x8
    3cb0:	7a448940 	ccmp	w10, #0x4, #0x0, hi  // hi = pmore
    3cb4:	540009a9 	b.ls	3de8 <sptMTTKRP+0x3b8>  // b.plast
    3cb8:	d1001042 	sub	x2, x2, #0x4
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3cbc:	d2800000 	mov	x0, #0x0                   	// #0
    3cc0:	8b0200a2 	add	x2, x5, x2
    3cc4:	d503201f 	nop
				scratch_data[r] *= times_mat_values[r];
    3cc8:	3ce06840 	ldr	q0, [x2, x0]
    3ccc:	3ce06821 	ldr	q1, [x1, x0]
    3cd0:	6e21dc00 	fmul	v0.4s, v0.4s, v1.4s
    3cd4:	3ca06820 	str	q0, [x1, x0]
			for(sptIndex r=0; r<R; ++r) {
    3cd8:	91004000 	add	x0, x0, #0x10
    3cdc:	eb04001f 	cmp	x0, x4
    3ce0:	54ffff41 	b.ne	3cc8 <sptMTTKRP+0x298>  // b.any
    3ce4:	6b1301bf 	cmp	w13, w19
    3ce8:	54000280 	b.eq	3d38 <sptMTTKRP+0x308>  // b.none
				scratch_data[r] *= times_mat_values[r];
    3cec:	8b0e0060 	add	x0, x3, x14
    3cf0:	bd400100 	ldr	s0, [x8]
    3cf4:	bc6078a1 	ldr	s1, [x5, x0, lsl #2]
    3cf8:	1e210800 	fmul	s0, s0, s1
    3cfc:	bd000100 	str	s0, [x8]
			for(sptIndex r=0; r<R; ++r) {
    3d00:	6b0f027f 	cmp	w19, w15
    3d04:	540001a9 	b.ls	3d38 <sptMTTKRP+0x308>  // b.plast
				scratch_data[r] *= times_mat_values[r];
    3d08:	8b100060 	add	x0, x3, x16
    3d0c:	bd400120 	ldr	s0, [x9]
    3d10:	bc6078a1 	ldr	s1, [x5, x0, lsl #2]
    3d14:	1e210800 	fmul	s0, s0, s1
    3d18:	bd000120 	str	s0, [x9]
			for(sptIndex r=0; r<R; ++r) {
    3d1c:	6b11027f 	cmp	w19, w17
    3d20:	540000c9 	b.ls	3d38 <sptMTTKRP+0x308>  // b.plast
				scratch_data[r] *= times_mat_values[r];
    3d24:	8b120063 	add	x3, x3, x18
    3d28:	bd400180 	ldr	s0, [x12]
    3d2c:	bc6378a1 	ldr	s1, [x5, x3, lsl #2]
    3d30:	1e210800 	fmul	s0, s0, s1
    3d34:	bd000180 	str	s0, [x12]
		for(sptIndex i=2; i<nmodes; ++i) {
    3d38:	910004c6 	add	x6, x6, #0x1
    3d3c:	6b0602bf 	cmp	w21, w6
    3d40:	54fffa08 	b.hi	3c80 <sptMTTKRP+0x250>  // b.pmore
		sptIndex tmp_mode = mode_i * stride;
    3d44:	b8676b63 	ldr	w3, [x27, x7]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3d48:	d2800000 	mov	x0, #0x0                   	// #0
		sptIndex tmp_mode = mode_i * stride;
    3d4c:	1b037ee3 	mul	w3, w23, w3
			mvals[tmp_mode + r] += scratch_data[r];
    3d50:	0b000062 	add	w2, w3, w0
    3d54:	bc607821 	ldr	s1, [x1, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3d58:	91000400 	add	x0, x0, #0x1
			mvals[tmp_mode + r] += scratch_data[r];
    3d5c:	bc627a80 	ldr	s0, [x20, x2, lsl #2]
    3d60:	1e212800 	fadd	s0, s0, s1
    3d64:	bc227a80 	str	s0, [x20, x2, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3d68:	6b00027f 	cmp	w19, w0
    3d6c:	54ffff28 	b.hi	3d50 <sptMTTKRP+0x320>  // b.pmore
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3d70:	f9403be0 	ldr	x0, [sp, #112]
    3d74:	910010e7 	add	x7, x7, #0x4
    3d78:	eb0000ff 	cmp	x7, x0
    3d7c:	54fff2a1 	b.ne	3bd0 <sptMTTKRP+0x1a0>  // b.any
	sptStopTimer(timer);
    3d80:	f94043e0 	ldr	x0, [sp, #128]
    3d84:	94000287 	bl	47a0 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    3d88:	f94043e0 	ldr	x0, [sp, #128]
    3d8c:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    3d90:	911c0021 	add	x1, x1, #0x700
    3d94:	940002c3 	bl	48a0 <sptPrintElapsedTime>
    3d98:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3d9c:	f94043e0 	ldr	x0, [sp, #128]
    3da0:	940002fc 	bl	4990 <sptFreeTimer>
	sptFreeValueVector(&scratch);
    3da4:	f9403fe0 	ldr	x0, [sp, #120]
    3da8:	97fff95f 	bl	2324 <sptFreeValueVector>
	printf("[Total time]: %lf\n", total_time);
    3dac:	1e604100 	fmov	d0, d8
    3db0:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3db4:	911ce000 	add	x0, x0, #0x738
    3db8:	97fff59a 	bl	1420 <printf@plt>
	printf("\n");
    3dbc:	52800140 	mov	w0, #0xa                   	// #10
    3dc0:	97fff5a4 	bl	1450 <putchar@plt>
	return 0;
    3dc4:	52800000 	mov	w0, #0x0                   	// #0
    3dc8:	fd4033e8 	ldr	d8, [sp, #96]
    3dcc:	a94153f3 	ldp	x19, x20, [sp, #16]
}
    3dd0:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    3dd4:	a94363f7 	ldp	x23, x24, [sp, #48]
    3dd8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3ddc:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3de0:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3de4:	d65f03c0 	ret
    3de8:	8b0308a3 	add	x3, x5, x3, lsl #2
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3dec:	d2800000 	mov	x0, #0x0                   	// #0
				scratch_data[r] *= times_mat_values[r];
    3df0:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    3df4:	bc607861 	ldr	s1, [x3, x0, lsl #2]
    3df8:	1e210800 	fmul	s0, s0, s1
    3dfc:	bc207820 	str	s0, [x1, x0, lsl #2]
			for(sptIndex r=0; r<R; ++r) {
    3e00:	91000400 	add	x0, x0, #0x1
    3e04:	6b00027f 	cmp	w19, w0
    3e08:	54ffff48 	b.hi	3df0 <sptMTTKRP+0x3c0>  // b.pmore
    3e0c:	17ffffcb 	b	3d38 <sptMTTKRP+0x308>
    3e10:	8b030bc3 	add	x3, x30, x3, lsl #2
		for(sptIndex r=0; r<R; ++r) {
    3e14:	d2800000 	mov	x0, #0x0                   	// #0
			scratch.data[r] = entry * times_mat_values[r];
    3e18:	bc607860 	ldr	s0, [x3, x0, lsl #2]
    3e1c:	1e220800 	fmul	s0, s0, s2
    3e20:	bc207820 	str	s0, [x1, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3e24:	91000400 	add	x0, x0, #0x1
    3e28:	6b00027f 	cmp	w19, w0
    3e2c:	54ffff68 	b.hi	3e18 <sptMTTKRP+0x3e8>  // b.pmore
    3e30:	17ffff90 	b	3c70 <sptMTTKRP+0x240>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3e34:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    3e38:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    3e3c:	911c6084 	add	x4, x4, #0x718
    3e40:	911b6042 	add	x2, x2, #0x6d8
    3e44:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    3e48:	52800823 	mov	w3, #0x41                  	// #65
    3e4c:	17ffff1c 	b	3abc <sptMTTKRP+0x8c>
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    3e50:	97fffe60 	bl	37d0 <sptMTTKRP_3D>
    3e54:	34ffe400 	cbz	w0, 3ad4 <sptMTTKRP+0xa4>
    3e58:	12800000 	mov	w0, #0xffffffff            	// #-1
    3e5c:	a90153f3 	stp	x19, x20, [sp, #16]
    3e60:	a90363f7 	stp	x23, x24, [sp, #48]
    3e64:	a9046bf9 	stp	x25, x26, [sp, #64]
    3e68:	a90573fb 	stp	x27, x28, [sp, #80]
    3e6c:	fd0033e8 	str	d8, [sp, #96]
    3e70:	97fff4dc 	bl	11e0 <exit@plt>
    3e74:	d503201f 	nop
    3e78:	d503201f 	nop
    3e7c:	d503201f 	nop

0000000000003e80 <sptOmpMTTKRP._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3e80:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    3e84:	910003fd 	mov	x29, sp
    3e88:	a9025bf5 	stp	x21, x22, [sp, #32]
    3e8c:	f9400c15 	ldr	x21, [x0, #24]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3e90:	b4001075 	cbz	x21, 409c <sptOmpMTTKRP._omp_fn.0+0x21c>
    3e94:	a90153f3 	stp	x19, x20, [sp, #16]
    3e98:	aa0003f4 	mov	x20, x0
    3e9c:	97fff521 	bl	1320 <omp_get_num_threads@plt>
    3ea0:	93407c13 	sxtw	x19, w0
    3ea4:	97fff4e7 	bl	1240 <omp_get_thread_num@plt>
    3ea8:	93407c16 	sxtw	x22, w0
    3eac:	9ad30aa1 	udiv	x1, x21, x19
    3eb0:	9b13d420 	msub	x0, x1, x19, x21
    3eb4:	eb0002df 	cmp	x22, x0
    3eb8:	54001003 	b.cc	40b8 <sptOmpMTTKRP._omp_fn.0+0x238>  // b.lo, b.ul, b.last
    3ebc:	9b160036 	madd	x22, x1, x22, x0
    3ec0:	8b160020 	add	x0, x1, x22
    3ec4:	f90043e0 	str	x0, [sp, #128]
    3ec8:	eb0002df 	cmp	x22, x0
    3ecc:	54000ee2 	b.cs	40a8 <sptOmpMTTKRP._omp_fn.0+0x228>  // b.hs, b.nlast
    3ed0:	9102e3e0 	add	x0, sp, #0xb8
    3ed4:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    3ed8:	b940429c 	ldr	w28, [x20, #64]
    3edc:	f9003fe0 	str	x0, [sp, #120]
    3ee0:	f9400280 	ldr	x0, [x20]
    3ee4:	f9004be0 	str	x0, [sp, #144]
    3ee8:	b9403a80 	ldr	w0, [x20, #56]
    3eec:	51000795 	sub	w21, w28, #0x1
    3ef0:	b90077e0 	str	w0, [sp, #116]
		sptValueVector scratch;  // Temporary array
		sptNewValueVector(&scratch, R, R);
    3ef4:	2a1c03e0 	mov	w0, w28
    3ef8:	f9004fe0 	str	x0, [sp, #152]
    3efc:	910006b5 	add	x21, x21, #0x1
#pragma omp parallel for schedule(static) num_threads(tk)
    3f00:	f9401280 	ldr	x0, [x20, #32]
    3f04:	f90047e0 	str	x0, [sp, #136]
    3f08:	f9401680 	ldr	x0, [x20, #40]
    3f0c:	a90363f7 	stp	x23, x24, [sp, #48]
    3f10:	d37ef6b5 	lsl	x21, x21, #2
    3f14:	a9046bf9 	stp	x25, x26, [sp, #64]
    3f18:	f90057e0 	str	x0, [sp, #168]
    3f1c:	a940e299 	ldp	x25, x24, [x20, #8]
    3f20:	fd0033e8 	str	d8, [sp, #96]
    3f24:	f9401a80 	ldr	x0, [x20, #48]
    3f28:	f90053e0 	str	x0, [sp, #160]
    3f2c:	b9403e94 	ldr	w20, [x20, #60]
		sptNewValueVector(&scratch, R, R);
    3f30:	f9404fe1 	ldr	x1, [sp, #152]
    3f34:	f9403ff3 	ldr	x19, [sp, #120]
    3f38:	aa0103e2 	mov	x2, x1
    3f3c:	aa1303e0 	mov	x0, x19
    3f40:	97fff84c 	bl	2070 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    3f44:	0f000400 	movi	v0.2s, #0x0
    3f48:	aa1303e0 	mov	x0, x19
    3f4c:	97fff871 	bl	2110 <sptConstantValueVector>

		sptIndex times_mat_index = mats_order[1];
		sptMatrix * times_mat = mats[times_mat_index];
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3f50:	f9404be1 	ldr	x1, [sp, #144]
    3f54:	d37ef6c8 	lsl	x8, x22, #2
		sptIndex times_mat_index = mats_order[1];
    3f58:	b9400700 	ldr	w0, [x24, #4]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3f5c:	f9401027 	ldr	x7, [x1, #32]
		sptIndex tmp_i = times_inds[x];
		sptValue const entry = vals[x];
    3f60:	f94047e1 	ldr	x1, [sp, #136]
		sptMatrix * times_mat = mats[times_mat_index];
    3f64:	f8605b22 	ldr	x2, [x25, w0, uxtw #3]
		sptValue const entry = vals[x];
    3f68:	bc767821 	ldr	s1, [x1, x22, lsl #2]
		sptIndex tmp_i = times_inds[x];
    3f6c:	52800301 	mov	w1, #0x18                  	// #24
    3f70:	9ba11c00 	umaddl	x0, w0, w1, x7
    3f74:	f9400800 	ldr	x0, [x0, #16]
    3f78:	b8767803 	ldr	w3, [x0, x22, lsl #2]
#pragma omp simd
		for(sptIndex r=0; r<R; ++r) {
    3f7c:	340007bc 	cbz	w28, 4070 <sptOmpMTTKRP._omp_fn.0+0x1f0>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    3f80:	1b037e83 	mul	w3, w20, w3
    3f84:	d2800000 	mov	x0, #0x0                   	// #0
    3f88:	f9400845 	ldr	x5, [x2, #16]
    3f8c:	f94067e1 	ldr	x1, [sp, #200]
    3f90:	0b000062 	add	w2, w3, w0
    3f94:	bc6278a0 	ldr	s0, [x5, x2, lsl #2]
    3f98:	1e210800 	fmul	s0, s0, s1
    3f9c:	bc207820 	str	s0, [x1, x0, lsl #2]
    3fa0:	91000400 	add	x0, x0, #0x1
    3fa4:	6b00039f 	cmp	w28, w0
    3fa8:	54ffff48 	b.hi	3f90 <sptOmpMTTKRP._omp_fn.0+0x110>  // b.pmore
		}

		for(sptIndex i=2; i<nmodes; ++i) {
    3fac:	b94077e0 	ldr	w0, [sp, #116]
			times_inds = X->inds[times_mat_index].data;
			tmp_i = times_inds[x];

#pragma omp simd
			for(sptIndex r=0; r<R; ++r) {
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3fb0:	d2800046 	mov	x6, #0x2                   	// #2
		for(sptIndex i=2; i<nmodes; ++i) {
    3fb4:	7100081f 	cmp	w0, #0x2
    3fb8:	54000309 	b.ls	4018 <sptOmpMTTKRP._omp_fn.0+0x198>  // b.plast
    3fbc:	d503201f 	nop
			times_mat_index = mats_order[i];
    3fc0:	b8667b02 	ldr	w2, [x24, x6, lsl #2]
			tmp_i = times_inds[x];
    3fc4:	52800303 	mov	w3, #0x18                  	// #24
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3fc8:	d2800000 	mov	x0, #0x0                   	// #0
			tmp_i = times_inds[x];
    3fcc:	9ba31c45 	umaddl	x5, w2, w3, x7
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3fd0:	f8625b23 	ldr	x3, [x25, w2, uxtw #3]
			tmp_i = times_inds[x];
    3fd4:	f94008a2 	ldr	x2, [x5, #16]
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3fd8:	f9400865 	ldr	x5, [x3, #16]
    3fdc:	b8686843 	ldr	w3, [x2, x8]
    3fe0:	1b037e83 	mul	w3, w20, w3
    3fe4:	d503201f 	nop
    3fe8:	0b000062 	add	w2, w3, w0
    3fec:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    3ff0:	bc6278a1 	ldr	s1, [x5, x2, lsl #2]
    3ff4:	1e210800 	fmul	s0, s0, s1
    3ff8:	bc207820 	str	s0, [x1, x0, lsl #2]
    3ffc:	91000400 	add	x0, x0, #0x1
    4000:	6b00039f 	cmp	w28, w0
    4004:	54ffff28 	b.hi	3fe8 <sptOmpMTTKRP._omp_fn.0+0x168>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    4008:	b94077e0 	ldr	w0, [sp, #116]
    400c:	910004c6 	add	x6, x6, #0x1
    4010:	6b06001f 	cmp	w0, w6
    4014:	54fffd68 	b.hi	3fc0 <sptOmpMTTKRP._omp_fn.0+0x140>  // b.pmore
			}
		}

		sptIndex const mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    4018:	f94057e0 	ldr	x0, [sp, #168]
#pragma omp parallel for schedule(static) num_threads(tk)
    401c:	d280001b 	mov	x27, #0x0                   	// #0
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    4020:	b8767813 	ldr	w19, [x0, x22, lsl #2]
    4024:	f94053e0 	ldr	x0, [sp, #160]
    4028:	1b137e93 	mul	w19, w20, w19
		for(sptIndex r=0; r<R; ++r) {
    402c:	8b130813 	add	x19, x0, x19, lsl #2
    4030:	14000002 	b	4038 <sptOmpMTTKRP._omp_fn.0+0x1b8>
    4034:	f94067e1 	ldr	x1, [sp, #200]
    4038:	8b130377 	add	x23, x27, x19
#pragma omp atomic update
			mvals_row[r] += scratch.data[r];
    403c:	bc7b6828 	ldr	s8, [x1, x27]
    4040:	b94002fa 	ldr	w26, [x23]
    4044:	1e270340 	fmov	s0, w26
    4048:	aa1703e2 	mov	x2, x23
    404c:	2a1a03e0 	mov	w0, w26
    4050:	1e202900 	fadd	s0, s8, s0
    4054:	1e260001 	fmov	w1, s0
    4058:	9400036a 	bl	4e00 <__aarch64_cas4_acq_rel>
    405c:	6b00035f 	cmp	w26, w0
    4060:	54000321 	b.ne	40c4 <sptOmpMTTKRP._omp_fn.0+0x244>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    4064:	9100137b 	add	x27, x27, #0x4
    4068:	eb15037f 	cmp	x27, x21
    406c:	54fffe41 	b.ne	4034 <sptOmpMTTKRP._omp_fn.0+0x1b4>  // b.any
		}

		sptFreeValueVector(&scratch);
    4070:	f9403fe0 	ldr	x0, [sp, #120]
    4074:	910006d6 	add	x22, x22, #0x1
    4078:	97fff8ab 	bl	2324 <sptFreeValueVector>
    407c:	f94043e0 	ldr	x0, [sp, #128]
    4080:	eb16001f 	cmp	x0, x22
    4084:	54fff568 	b.hi	3f30 <sptOmpMTTKRP._omp_fn.0+0xb0>  // b.pmore
    4088:	a94153f3 	ldp	x19, x20, [sp, #16]
    408c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4090:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4094:	a94573fb 	ldp	x27, x28, [sp, #80]
    4098:	fd4033e8 	ldr	d8, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    409c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    40a0:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    40a4:	d65f03c0 	ret
    40a8:	a94153f3 	ldp	x19, x20, [sp, #16]
    40ac:	a9425bf5 	ldp	x21, x22, [sp, #32]
    40b0:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    40b4:	d65f03c0 	ret
    40b8:	91000421 	add	x1, x1, #0x1
    40bc:	d2800000 	mov	x0, #0x0                   	// #0
    40c0:	17ffff7f 	b	3ebc <sptOmpMTTKRP._omp_fn.0+0x3c>
			mvals_row[r] += scratch.data[r];
    40c4:	2a0003fa 	mov	w26, w0
    40c8:	17ffffdf 	b	4044 <sptOmpMTTKRP._omp_fn.0+0x1c4>
    40cc:	d503201f 	nop

00000000000040d0 <sptOmpMTTKRP_3D._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    40d0:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    40d4:	910003fd 	mov	x29, sp
    40d8:	a9025bf5 	stp	x21, x22, [sp, #32]
    40dc:	f9400016 	ldr	x22, [x0]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    40e0:	b40009f6 	cbz	x22, 421c <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
    40e4:	a90153f3 	stp	x19, x20, [sp, #16]
    40e8:	aa0003f3 	mov	x19, x0
    40ec:	a90363f7 	stp	x23, x24, [sp, #48]
    40f0:	97fff48c 	bl	1320 <omp_get_num_threads@plt>
    40f4:	93407c14 	sxtw	x20, w0
    40f8:	97fff452 	bl	1240 <omp_get_thread_num@plt>
    40fc:	93407c15 	sxtw	x21, w0
    4100:	9ad40ad7 	udiv	x23, x22, x20
    4104:	9b14dae0 	msub	x0, x23, x20, x22
    4108:	eb0002bf 	cmp	x21, x0
    410c:	54000983 	b.cc	423c <sptOmpMTTKRP_3D._omp_fn.0+0x16c>  // b.lo, b.ul, b.last
    4110:	9b1502f5 	madd	x21, x23, x21, x0
    4114:	8b1502e0 	add	x0, x23, x21
    4118:	f9003be0 	str	x0, [sp, #112]
    411c:	eb0002bf 	cmp	x21, x0
    4120:	54000842 	b.cs	4228 <sptOmpMTTKRP_3D._omp_fn.0+0x158>  // b.hs, b.nlast
    4124:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    4128:	a940ee61 	ldp	x1, x27, [x19, #8]
    412c:	f9004fe1 	str	x1, [sp, #152]
    4130:	f9400e61 	ldr	x1, [x19, #24]
    4134:	f90047e1 	str	x1, [sp, #136]
    4138:	29480276 	ldp	w22, w0, [x19, #64]
    413c:	b90097e0 	str	w0, [sp, #148]
    4140:	f9401661 	ldr	x1, [x19, #40]
    4144:	f90043e1 	str	x1, [sp, #128]
    4148:	f9401e61 	ldr	x1, [x19, #56]
    414c:	f9003fe1 	str	x1, [sp, #120]
    4150:	f9401278 	ldr	x24, [x19, #32]
    4154:	f9401a7c 	ldr	x28, [x19, #48]
    4158:	34000780 	cbz	w0, 4248 <sptOmpMTTKRP_3D._omp_fn.0+0x178>
    415c:	a9046bf9 	stp	x25, x26, [sp, #64]
    4160:	6d0627e8 	stp	d8, d9, [sp, #96]
    4164:	d503201f 	nop
		sptIndex mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    4168:	b8757b62 	ldr	w2, [x27, x21, lsl #2]
		sptIndex tmp_i_2 = times_inds_2[x];
		sptValue entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    416c:	f9403fe0 	ldr	x0, [sp, #120]
    4170:	1b027ec2 	mul	w2, w22, w2
    4174:	b8757813 	ldr	w19, [x0, x21, lsl #2]
    4178:	f94043e0 	ldr	x0, [sp, #128]
    417c:	1b137ed3 	mul	w19, w22, w19
    4180:	b8757814 	ldr	w20, [x0, x21, lsl #2]
		sptValue entry = vals[x];
    4184:	f9404fe0 	ldr	x0, [sp, #152]
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4188:	2a1303f9 	mov	w25, w19
    418c:	1b147ed4 	mul	w20, w22, w20
		sptValue entry = vals[x];
    4190:	bc757809 	ldr	s9, [x0, x21, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    4194:	b94097e0 	ldr	w0, [sp, #148]
    4198:	4b130294 	sub	w20, w20, w19
    419c:	0b130013 	add	w19, w0, w19
    41a0:	f94047e0 	ldr	x0, [sp, #136]
    41a4:	8b020817 	add	x23, x0, x2, lsl #2
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    41a8:	f9400b04 	ldr	x4, [x24, #16]
    41ac:	0b190280 	add	w0, w20, w25
    41b0:	f9400b81 	ldr	x1, [x28, #16]
    41b4:	bc607888 	ldr	s8, [x4, x0, lsl #2]
    41b8:	bc795820 	ldr	s0, [x1, w25, uxtw #2]
    41bc:	1e200908 	fmul	s8, s8, s0
    41c0:	1e290908 	fmul	s8, s8, s9
    41c4:	b94002fa 	ldr	w26, [x23]
    41c8:	1e270340 	fmov	s0, w26
    41cc:	aa1703e2 	mov	x2, x23
    41d0:	2a1a03e0 	mov	w0, w26
    41d4:	1e202900 	fadd	s0, s8, s0
    41d8:	1e260001 	fmov	w1, s0
    41dc:	94000309 	bl	4e00 <__aarch64_cas4_acq_rel>
    41e0:	6b00035f 	cmp	w26, w0
    41e4:	540003a1 	b.ne	4258 <sptOmpMTTKRP_3D._omp_fn.0+0x188>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    41e8:	11000739 	add	w25, w25, #0x1
    41ec:	910012f7 	add	x23, x23, #0x4
    41f0:	6b19027f 	cmp	w19, w25
    41f4:	54fffda1 	b.ne	41a8 <sptOmpMTTKRP_3D._omp_fn.0+0xd8>  // b.any
    41f8:	f9403be0 	ldr	x0, [sp, #112]
    41fc:	910006b5 	add	x21, x21, #0x1
    4200:	eb15001f 	cmp	x0, x21
    4204:	54fffb21 	b.ne	4168 <sptOmpMTTKRP_3D._omp_fn.0+0x98>  // b.any
    4208:	a94153f3 	ldp	x19, x20, [sp, #16]
    420c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4210:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4214:	a94573fb 	ldp	x27, x28, [sp, #80]
    4218:	6d4627e8 	ldp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    421c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4220:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4224:	d65f03c0 	ret
    4228:	a94153f3 	ldp	x19, x20, [sp, #16]
    422c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4230:	a94363f7 	ldp	x23, x24, [sp, #48]
    4234:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4238:	d65f03c0 	ret
    423c:	910006f7 	add	x23, x23, #0x1
    4240:	d2800000 	mov	x0, #0x0                   	// #0
    4244:	17ffffb3 	b	4110 <sptOmpMTTKRP_3D._omp_fn.0+0x40>
    4248:	a94153f3 	ldp	x19, x20, [sp, #16]
    424c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4250:	a94573fb 	ldp	x27, x28, [sp, #80]
    4254:	17fffff2 	b	421c <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4258:	2a0003fa 	mov	w26, w0
    425c:	17ffffdb 	b	41c8 <sptOmpMTTKRP_3D._omp_fn.0+0xf8>

0000000000004260 <sptOmpMTTKRP_3D>:
{
    4260:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    4264:	910003fd 	mov	x29, sp
    4268:	a90153f3 	stp	x19, x20, [sp, #16]
    426c:	aa0003f4 	mov	x20, x0
    4270:	aa0103f3 	mov	x19, x1
	sptIndex const stride = mats[0]->stride;
    4274:	f9400020 	ldr	x0, [x1]
{
    4278:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptAssert(nmodes ==3);
    427c:	b9400281 	ldr	w1, [x20]
{
    4280:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex const stride = mats[0]->stride;
    4284:	b9400c16 	ldr	w22, [x0, #12]
	sptNnzIndex const nnz = X->nnz;
    4288:	a9416285 	ldp	x5, x24, [x20, #16]
	sptValue const * const restrict vals = X->values.data;
    428c:	f9401e97 	ldr	x23, [x20, #56]
	sptAssert(nmodes ==3);
    4290:	a9046bf9 	stp	x25, x26, [sp, #64]
    4294:	71000c3f 	cmp	w1, #0x3
    4298:	54000e61 	b.ne	4464 <sptOmpMTTKRP_3D+0x204>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    429c:	f9400e66 	ldr	x6, [x19, #24]
    42a0:	aa0203fa 	mov	x26, x2
    42a4:	b9400402 	ldr	w2, [x0, #4]
    42a8:	b94004c1 	ldr	w1, [x6, #4]
    42ac:	6b02003f 	cmp	w1, w2
    42b0:	54000ae1 	b.ne	440c <sptOmpMTTKRP_3D+0x1ac>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    42b4:	b9400002 	ldr	w2, [x0]
    42b8:	b94000a0 	ldr	w0, [x5]
    42bc:	6b00005f 	cmp	w2, w0
    42c0:	54000bc1 	b.ne	4438 <sptOmpMTTKRP_3D+0x1d8>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    42c4:	f9400660 	ldr	x0, [x19, #8]
    42c8:	b9400402 	ldr	w2, [x0, #4]
    42cc:	6b02003f 	cmp	w1, w2
    42d0:	540009e1 	b.ne	440c <sptOmpMTTKRP_3D+0x1ac>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    42d4:	b9400002 	ldr	w2, [x0]
    42d8:	b94004a0 	ldr	w0, [x5, #4]
    42dc:	6b00005f 	cmp	w2, w0
    42e0:	54000ac1 	b.ne	4438 <sptOmpMTTKRP_3D+0x1d8>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    42e4:	f9400a60 	ldr	x0, [x19, #16]
    42e8:	b9400402 	ldr	w2, [x0, #4]
    42ec:	6b01005f 	cmp	w2, w1
    42f0:	540008e1 	b.ne	440c <sptOmpMTTKRP_3D+0x1ac>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    42f4:	b9400001 	ldr	w1, [x0]
    42f8:	b94008a0 	ldr	w0, [x5, #8]
    42fc:	6b00003f 	cmp	w1, w0
    4300:	540009c1 	b.ne	4438 <sptOmpMTTKRP_3D+0x1d8>  // b.any
	sptIndex const tmpI = mats[mode]->nrows;
    4304:	f8635a60 	ldr	x0, [x19, w3, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    4308:	52800315 	mov	w21, #0x18                  	// #24
    430c:	f9401282 	ldr	x2, [x20, #32]
    4310:	a90573fb 	stp	x27, x28, [sp, #80]
    4314:	2a0403f9 	mov	w25, w4
	sptIndex const R = mats[mode]->ncols;
    4318:	29407005 	ldp	w5, w28, [x0]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    431c:	9bb50863 	umaddl	x3, w3, w21, x2
	sptValue * const restrict mvals = mats[nmodes]->values;
    4320:	f94008db 	ldr	x27, [x6, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4324:	52800001 	mov	w1, #0x0                   	// #0
    4328:	1b057ec5 	mul	w5, w22, w5
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    432c:	f9400867 	ldr	x7, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4330:	aa1b03e0 	mov	x0, x27
    4334:	d37ef4a2 	lsl	x2, x5, #2
    4338:	fd0033e8 	str	d8, [sp, #96]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    433c:	f9003fe7 	str	x7, [sp, #120]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4340:	97fff3e4 	bl	12d0 <memset@plt>
	sptIndex times_mat_index_2 = mats_order[2];
    4344:	29409743 	ldp	w3, w5, [x26, #4]
	sptNewTimer(&timer, 0);
    4348:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    434c:	f9401286 	ldr	x6, [x20, #32]
	sptNewTimer(&timer, 0);
    4350:	910203e0 	add	x0, sp, #0x80
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    4354:	f8635a64 	ldr	x4, [x19, w3, uxtw #3]
    4358:	f9003be4 	str	x4, [sp, #112]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    435c:	9bb51863 	umaddl	x3, w3, w21, x6
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    4360:	9bb518b5 	umaddl	x21, w5, w21, x6
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    4364:	f8655a74 	ldr	x20, [x19, w5, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4368:	f940087a 	ldr	x26, [x3, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    436c:	f9400ab3 	ldr	x19, [x21, #16]
	sptNewTimer(&timer, 0);
    4370:	940000bc 	bl	4660 <sptNewTimer>
	sptStartTimer(timer);
    4374:	f94043e0 	ldr	x0, [sp, #128]
    4378:	940000da 	bl	46e0 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    437c:	a908dff8 	stp	x24, x23, [sp, #136]
    4380:	2a1903e2 	mov	w2, w25
    4384:	a9471fe4 	ldp	x4, x7, [sp, #112]
    4388:	52800003 	mov	w3, #0x0                   	// #0
    438c:	910223e1 	add	x1, sp, #0x88
    4390:	90000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0x180>
    4394:	91034000 	add	x0, x0, #0xd0
    4398:	a909efe7 	stp	x7, x27, [sp, #152]
    439c:	a90aebe4 	stp	x4, x26, [sp, #168]
    43a0:	a90bcff4 	stp	x20, x19, [sp, #184]
    43a4:	291973f6 	stp	w22, w28, [sp, #200]
    43a8:	97fff426 	bl	1440 <GOMP_parallel@plt>
		}
	}
	sptStopTimer(timer);
    43ac:	f94043e0 	ldr	x0, [sp, #128]
    43b0:	940000fc 	bl	47a0 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    43b4:	f94043e0 	ldr	x0, [sp, #128]
    43b8:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    43bc:	911de021 	add	x1, x1, #0x778
    43c0:	94000138 	bl	48a0 <sptPrintElapsedTime>
    43c4:	1e604008 	fmov	d8, d0

	sptFreeTimer(timer);
    43c8:	f94043e0 	ldr	x0, [sp, #128]
    43cc:	94000171 	bl	4990 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    43d0:	1e604100 	fmov	d0, d8
    43d4:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    43d8:	911ce000 	add	x0, x0, #0x738
    43dc:	97fff411 	bl	1420 <printf@plt>
	printf("\n");
    43e0:	52800140 	mov	w0, #0xa                   	// #10
    43e4:	97fff41b 	bl	1450 <putchar@plt>

	return 0;
    43e8:	a94573fb 	ldp	x27, x28, [sp, #80]
    43ec:	52800000 	mov	w0, #0x0                   	// #0
    43f0:	fd4033e8 	ldr	d8, [sp, #96]
}
    43f4:	a94153f3 	ldp	x19, x20, [sp, #16]
    43f8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    43fc:	a94363f7 	ldp	x23, x24, [sp, #48]
    4400:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4404:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4408:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    440c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    4410:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4414:	911ac084 	add	x4, x4, #0x6b0
    4418:	911d4042 	add	x2, x2, #0x750
    441c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    4420:	528012c3 	mov	w3, #0x96                  	// #150
    4424:	911de000 	add	x0, x0, #0x778
    4428:	52800041 	mov	w1, #0x2                   	// #2
    442c:	940001d9 	bl	4b90 <spt_ComplainError>
    4430:	52800040 	mov	w0, #0x2                   	// #2
    4434:	17fffff0 	b	43f4 <sptOmpMTTKRP_3D+0x194>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4438:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    443c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4440:	911c6084 	add	x4, x4, #0x718
    4444:	911d4042 	add	x2, x2, #0x750
    4448:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    444c:	52801323 	mov	w3, #0x99                  	// #153
    4450:	911de000 	add	x0, x0, #0x778
    4454:	52800041 	mov	w1, #0x2                   	// #2
    4458:	940001ce 	bl	4b90 <spt_ComplainError>
    445c:	52800040 	mov	w0, #0x2                   	// #2
    4460:	17ffffe5 	b	43f4 <sptOmpMTTKRP_3D+0x194>
	sptAssert(nmodes ==3);
    4464:	12800000 	mov	w0, #0xffffffff            	// #-1
    4468:	a90573fb 	stp	x27, x28, [sp, #80]
    446c:	fd0033e8 	str	d8, [sp, #96]
    4470:	97fff35c 	bl	11e0 <exit@plt>

0000000000004474 <sptOmpMTTKRP>:
{
    4474:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    4478:	910003fd 	mov	x29, sp
    447c:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const nmodes = X->nmodes;
    4480:	b9400014 	ldr	w20, [x0]
	if(nmodes == 3) {
    4484:	71000e9f 	cmp	w20, #0x3
    4488:	54000da0 	b.eq	463c <sptOmpMTTKRP+0x1c8>  // b.none
	sptNnzIndex const nnz = X->nnz;
    448c:	a90363f7 	stp	x23, x24, [sp, #48]
    4490:	aa0003f7 	mov	x23, x0
	sptIndex const * const ndims = X->ndims;
    4494:	aa0103f3 	mov	x19, x1
    4498:	a9025bf5 	stp	x21, x22, [sp, #32]
    449c:	2a0303e9 	mov	w9, w3
    44a0:	aa0203f6 	mov	x22, x2
    44a4:	2a0403f5 	mov	w21, w4
	sptValue * const restrict mvals = mats[nmodes]->values;
    44a8:	f8745820 	ldr	x0, [x1, w20, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    44ac:	f9400ae8 	ldr	x8, [x23, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    44b0:	340003d4 	cbz	w20, 4528 <sptOmpMTTKRP+0xb4>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    44b4:	b9400401 	ldr	w1, [x0, #4]
    44b8:	d2800005 	mov	x5, #0x0                   	// #0
    44bc:	14000008 	b	44dc <sptOmpMTTKRP+0x68>
		if(mats[i]->nrows != ndims[i]) {
    44c0:	b94000c7 	ldr	w7, [x6]
    44c4:	b8657906 	ldr	w6, [x8, x5, lsl #2]
    44c8:	6b0600ff 	cmp	w7, w6
    44cc:	54000aa1 	b.ne	4620 <sptOmpMTTKRP+0x1ac>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    44d0:	910004a5 	add	x5, x5, #0x1
    44d4:	6b05029f 	cmp	w20, w5
    44d8:	54000289 	b.ls	4528 <sptOmpMTTKRP+0xb4>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    44dc:	f8657a66 	ldr	x6, [x19, x5, lsl #3]
    44e0:	b94004c7 	ldr	w7, [x6, #4]
    44e4:	6b0100ff 	cmp	w7, w1
    44e8:	54fffec0 	b.eq	44c0 <sptOmpMTTKRP+0x4c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    44ec:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    44f0:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    44f4:	911ac084 	add	x4, x4, #0x6b0
    44f8:	911d4042 	add	x2, x2, #0x750
    44fc:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    4500:	52800823 	mov	w3, #0x41                  	// #65
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4504:	911de000 	add	x0, x0, #0x778
    4508:	52800041 	mov	w1, #0x2                   	// #2
    450c:	940001a1 	bl	4b90 <spt_ComplainError>
    4510:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4514:	52800040 	mov	w0, #0x2                   	// #2
    4518:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    451c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4520:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4524:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    4528:	f8695a64 	ldr	x4, [x19, w9, uxtw #3]
    452c:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4530:	52800305 	mov	w5, #0x18                  	// #24
	sptIndex const stride = mats[0]->stride;
    4534:	f9400262 	ldr	x2, [x19]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4538:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    453c:	b940049a 	ldr	w26, [x4, #4]
	sptValue * const restrict mvals = mats[nmodes]->values;
    4540:	f9400819 	ldr	x25, [x0, #16]
    4544:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex const stride = mats[0]->stride;
    4548:	b9400c58 	ldr	w24, [x2, #12]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    454c:	b9400082 	ldr	w2, [x4]
    4550:	aa1903e0 	mov	x0, x25
	sptIndex const * const mode_ind = X->inds[mode].data;
    4554:	a9418ee4 	ldp	x4, x3, [x23, #24]
	sptNnzIndex const nnz = X->nnz;
    4558:	f9003fe4 	str	x4, [sp, #120]
	sptValue const * const vals = X->values.data;
    455c:	f9401efc 	ldr	x28, [x23, #56]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4560:	1b027f02 	mul	w2, w24, w2
	sptIndex const * const mode_ind = X->inds[mode].data;
    4564:	9ba50d23 	umaddl	x3, w9, w5, x3
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4568:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const mode_ind = X->inds[mode].data;
    456c:	f940087b 	ldr	x27, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4570:	fd0033e8 	str	d8, [sp, #96]
    4574:	97fff357 	bl	12d0 <memset@plt>
	sptNewTimer(&timer, 0);
    4578:	52800001 	mov	w1, #0x0                   	// #0
    457c:	910203e0 	add	x0, sp, #0x80
    4580:	94000038 	bl	4660 <sptNewTimer>
	sptStartTimer(timer);
    4584:	f94043e0 	ldr	x0, [sp, #128]
    4588:	94000056 	bl	46e0 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    458c:	a908cff7 	stp	x23, x19, [sp, #136]
    4590:	2a1503e2 	mov	w2, w21
    4594:	f9403fe4 	ldr	x4, [sp, #120]
    4598:	52800003 	mov	w3, #0x0                   	// #0
    459c:	910223e1 	add	x1, sp, #0x88
    45a0:	f0ffffe0 	adrp	x0, 3000 <sptLoadSparseTensor+0x8bc>
    45a4:	913a0000 	add	x0, x0, #0xe80
    45a8:	a90993f6 	stp	x22, x4, [sp, #152]
    45ac:	a90aeffc 	stp	x28, x27, [sp, #168]
    45b0:	f9005ff9 	str	x25, [sp, #184]
    45b4:	291863f4 	stp	w20, w24, [sp, #192]
    45b8:	b900cbfa 	str	w26, [sp, #200]
    45bc:	97fff3a1 	bl	1440 <GOMP_parallel@plt>
	sptStopTimer(timer);
    45c0:	f94043e0 	ldr	x0, [sp, #128]
    45c4:	94000077 	bl	47a0 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    45c8:	f94043e0 	ldr	x0, [sp, #128]
    45cc:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    45d0:	911de021 	add	x1, x1, #0x778
    45d4:	940000b3 	bl	48a0 <sptPrintElapsedTime>
    45d8:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    45dc:	f94043e0 	ldr	x0, [sp, #128]
    45e0:	940000ec 	bl	4990 <sptFreeTimer>
	printf("[Total time]: %lf\n", total_time);
    45e4:	1e604100 	fmov	d0, d8
    45e8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    45ec:	911ce000 	add	x0, x0, #0x738
    45f0:	97fff38c 	bl	1420 <printf@plt>
	printf("\n");
    45f4:	52800140 	mov	w0, #0xa                   	// #10
    45f8:	97fff396 	bl	1450 <putchar@plt>
	return 0;
    45fc:	52800000 	mov	w0, #0x0                   	// #0
    4600:	fd4033e8 	ldr	d8, [sp, #96]
}
    4604:	a94153f3 	ldp	x19, x20, [sp, #16]
	return 0;
    4608:	a9425bf5 	ldp	x21, x22, [sp, #32]
    460c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4610:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4614:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    4618:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    461c:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4620:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    4624:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4628:	911c6084 	add	x4, x4, #0x718
    462c:	911d4042 	add	x2, x2, #0x750
    4630:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    4634:	52800883 	mov	w3, #0x44                  	// #68
    4638:	17ffffb3 	b	4504 <sptOmpMTTKRP+0x90>
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    463c:	97ffff09 	bl	4260 <sptOmpMTTKRP_3D>
    4640:	34fff6e0 	cbz	w0, 451c <sptOmpMTTKRP+0xa8>
    4644:	12800000 	mov	w0, #0xffffffff            	// #-1
    4648:	a9025bf5 	stp	x21, x22, [sp, #32]
    464c:	a90363f7 	stp	x23, x24, [sp, #48]
    4650:	a9046bf9 	stp	x25, x26, [sp, #64]
    4654:	a90573fb 	stp	x27, x28, [sp, #80]
    4658:	fd0033e8 	str	d8, [sp, #96]
    465c:	97fff2e1 	bl	11e0 <exit@plt>

0000000000004660 <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    4660:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4664:	910003fd 	mov	x29, sp
    4668:	a90153f3 	stp	x19, x20, [sp, #16]
    466c:	aa0003f4 	mov	x20, x0
    4670:	2a0103f3 	mov	w19, w1
	*timer = (sptTimer) malloc(sizeof **timer);
    4674:	d2800500 	mov	x0, #0x28                  	// #40
    4678:	97fff302 	bl	1280 <malloc@plt>
	(*timer)->use_cuda = use_cuda;
    467c:	b9000013 	str	w19, [x0]
	*timer = (sptTimer) malloc(sizeof **timer);
    4680:	f9000280 	str	x0, [x20]
	if(use_cuda) {
    4684:	350000b3 	cbnz	w19, 4698 <sptNewTimer+0x38>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	return 0;
}
    4688:	2a1303e0 	mov	w0, w19
    468c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4690:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4694:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4698:	52800073 	mov	w19, #0x3                   	// #3
    469c:	52800061 	mov	w1, #0x3                   	// #3
    46a0:	72a00053 	movk	w19, #0x2, lsl #16
    46a4:	72a00041 	movk	w1, #0x2, lsl #16
    46a8:	528004e3 	mov	w3, #0x27                  	// #39
    46ac:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    46b0:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    46b4:	911e4084 	add	x4, x4, #0x790
    46b8:	911ee042 	add	x2, x2, #0x7b8
    46bc:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    46c0:	911f8000 	add	x0, x0, #0x7e0
    46c4:	94000133 	bl	4b90 <spt_ComplainError>
}
    46c8:	2a1303e0 	mov	w0, w19
    46cc:	a94153f3 	ldp	x19, x20, [sp, #16]
    46d0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    46d4:	d65f03c0 	ret
    46d8:	d503201f 	nop
    46dc:	d503201f 	nop

00000000000046e0 <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    46e0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    46e4:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    46e8:	b9400001 	ldr	w1, [x0]
    46ec:	350000e1 	cbnz	w1, 4708 <sptStartTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    46f0:	91002001 	add	x1, x0, #0x8
    46f4:	52800020 	mov	w0, #0x1                   	// #1
    46f8:	97fff2c6 	bl	1210 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    46fc:	35000220 	cbnz	w0, 4740 <sptStartTimer+0x60>
	}
	return 0;
}
    4700:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4704:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4708:	52800061 	mov	w1, #0x3                   	// #3
    470c:	528005e3 	mov	w3, #0x2f                  	// #47
    4710:	72a00041 	movk	w1, #0x2, lsl #16
    4714:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    4718:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    471c:	911e4084 	add	x4, x4, #0x790
    4720:	911ee042 	add	x2, x2, #0x7b8
    4724:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    4728:	911f8000 	add	x0, x0, #0x7e0
    472c:	94000119 	bl	4b90 <spt_ComplainError>
    4730:	52800060 	mov	w0, #0x3                   	// #3
    4734:	72a00040 	movk	w0, #0x2, lsl #16
}
    4738:	a8c27bfd 	ldp	x29, x30, [sp], #32
    473c:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    4740:	a90153f3 	stp	x19, x20, [sp, #16]
    4744:	97fff33b 	bl	1430 <__errno_location@plt>
    4748:	aa0003f3 	mov	x19, x0
	return 0;
    474c:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    4750:	b9400274 	ldr	w20, [x19]
    4754:	3140429f 	cmn	w20, #0x10, lsl #12
    4758:	54000061 	b.ne	4764 <sptStartTimer+0x84>  // b.any
    475c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4760:	17ffffe8 	b	4700 <sptStartTimer+0x20>
    4764:	2a1403e0 	mov	w0, w20
    4768:	97fff2f2 	bl	1330 <strerror@plt>
    476c:	11404281 	add	w1, w20, #0x10, lsl #12
    4770:	aa0003e4 	mov	x4, x0
    4774:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4778:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    477c:	911ee042 	add	x2, x2, #0x7b8
    4780:	911f8000 	add	x0, x0, #0x7e0
    4784:	52800643 	mov	w3, #0x32                  	// #50
    4788:	94000102 	bl	4b90 <spt_ComplainError>
    478c:	b9400260 	ldr	w0, [x19]
    4790:	a94153f3 	ldp	x19, x20, [sp, #16]
    4794:	11404000 	add	w0, w0, #0x10, lsl #12
    4798:	17ffffda 	b	4700 <sptStartTimer+0x20>
    479c:	d503201f 	nop

00000000000047a0 <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    47a0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    47a4:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    47a8:	b9400001 	ldr	w1, [x0]
    47ac:	350000e1 	cbnz	w1, 47c8 <sptStopTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    47b0:	91006001 	add	x1, x0, #0x18
    47b4:	52800020 	mov	w0, #0x1                   	// #1
    47b8:	97fff296 	bl	1210 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    47bc:	35000220 	cbnz	w0, 4800 <sptStopTimer+0x60>
	}
	return 0;
}
    47c0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    47c4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    47c8:	52800061 	mov	w1, #0x3                   	// #3
    47cc:	52800743 	mov	w3, #0x3a                  	// #58
    47d0:	72a00041 	movk	w1, #0x2, lsl #16
    47d4:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    47d8:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    47dc:	911e4084 	add	x4, x4, #0x790
    47e0:	911ee042 	add	x2, x2, #0x7b8
    47e4:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    47e8:	911f8000 	add	x0, x0, #0x7e0
    47ec:	940000e9 	bl	4b90 <spt_ComplainError>
    47f0:	52800060 	mov	w0, #0x3                   	// #3
    47f4:	72a00040 	movk	w0, #0x2, lsl #16
}
    47f8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    47fc:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    4800:	a90153f3 	stp	x19, x20, [sp, #16]
    4804:	97fff30b 	bl	1430 <__errno_location@plt>
    4808:	aa0003f3 	mov	x19, x0
	return 0;
    480c:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    4810:	b9400274 	ldr	w20, [x19]
    4814:	3140429f 	cmn	w20, #0x10, lsl #12
    4818:	54000061 	b.ne	4824 <sptStopTimer+0x84>  // b.any
    481c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4820:	17ffffe8 	b	47c0 <sptStopTimer+0x20>
    4824:	2a1403e0 	mov	w0, w20
    4828:	97fff2c2 	bl	1330 <strerror@plt>
    482c:	11404281 	add	w1, w20, #0x10, lsl #12
    4830:	aa0003e4 	mov	x4, x0
    4834:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4838:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    483c:	911ee042 	add	x2, x2, #0x7b8
    4840:	911f8000 	add	x0, x0, #0x7e0
    4844:	528007a3 	mov	w3, #0x3d                  	// #61
    4848:	940000d2 	bl	4b90 <spt_ComplainError>
    484c:	b9400260 	ldr	w0, [x19]
    4850:	a94153f3 	ldp	x19, x20, [sp, #16]
    4854:	11404000 	add	w0, w0, #0x10, lsl #12
    4858:	17ffffda 	b	47c0 <sptStopTimer+0x20>
    485c:	d503201f 	nop

0000000000004860 <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
	if(timer->use_cuda) {
    4860:	b9400001 	ldr	w1, [x0]
    4864:	35000161 	cbnz	w1, 4890 <sptElapsedTime+0x30>
		return NAN;
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4868:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    486c:	a9408c04 	ldp	x4, x3, [x0, #8]
    4870:	fd419422 	ldr	d2, [x1, #808]
    4874:	a9418801 	ldp	x1, x2, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4878:	cb040020 	sub	x0, x1, x4
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    487c:	cb030041 	sub	x1, x2, x3
    4880:	9e620021 	scvtf	d1, x1
    4884:	9e620000 	scvtf	d0, x0
    4888:	1f420020 	fmadd	d0, d1, d2, d0
	}
}
    488c:	d65f03c0 	ret
		return NAN;
    4890:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4894:	9e670000 	fmov	d0, x0
}
    4898:	d65f03c0 	ret
    489c:	d503201f 	nop

00000000000048a0 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    48a0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    48a4:	aa0103e2 	mov	x2, x1
    48a8:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    48ac:	b9400001 	ldr	w1, [x0]
double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    48b0:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    48b4:	350002a1 	cbnz	w1, 4908 <sptPrintElapsedTime+0x68>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    48b8:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    48bc:	a9409005 	ldp	x5, x4, [x0, #8]
    48c0:	fd419421 	ldr	d1, [x1, #808]
    48c4:	a9418403 	ldp	x3, x1, [x0, #24]
    48c8:	cb040021 	sub	x1, x1, x4
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    48cc:	cb050060 	sub	x0, x3, x5
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    48d0:	9e620020 	scvtf	d0, x1
    48d4:	9e620008 	scvtf	d8, x0
    48d8:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer);
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    48dc:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    48e0:	1e604100 	fmov	d0, d8
    48e4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    48e8:	911fc021 	add	x1, x1, #0x7f0
    48ec:	f947e000 	ldr	x0, [x0, #4032]
    48f0:	f9400000 	ldr	x0, [x0]
    48f4:	97fff2db 	bl	1460 <fprintf@plt>
	return elapsed_time;
}
    48f8:	1e604100 	fmov	d0, d8
    48fc:	fd400be8 	ldr	d8, [sp, #16]
    4900:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4904:	d65f03c0 	ret
		return NAN;
    4908:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    490c:	9e670008 	fmov	d8, x0
    4910:	17fffff3 	b	48dc <sptPrintElapsedTime+0x3c>

0000000000004914 <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4914:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4918:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    491c:	b9400003 	ldr	w3, [x0]
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4920:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4924:	350002e3 	cbnz	w3, 4980 <sptPrintAverageElapsedTime+0x6c>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4928:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x120>
    492c:	a9409406 	ldp	x6, x5, [x0, #8]
    4930:	fd419461 	ldr	d1, [x3, #808]
    4934:	a9419003 	ldp	x3, x4, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4938:	cb060060 	sub	x0, x3, x6
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    493c:	cb050083 	sub	x3, x4, x5
    4940:	9e620060 	scvtf	d0, x3
    4944:	9e620008 	scvtf	d8, x0
    4948:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer) / niters;
    494c:	1e620020 	scvtf	d0, w1
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4950:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    4954:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    4958:	911fc021 	add	x1, x1, #0x7f0
    495c:	f947e000 	ldr	x0, [x0, #4032]
	double elapsed_time = sptElapsedTime(timer) / niters;
    4960:	1e601908 	fdiv	d8, d8, d0
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4964:	f9400000 	ldr	x0, [x0]
    4968:	1e604100 	fmov	d0, d8
    496c:	97fff2bd 	bl	1460 <fprintf@plt>
	return elapsed_time;
}
    4970:	1e604100 	fmov	d0, d8
    4974:	fd400be8 	ldr	d8, [sp, #16]
    4978:	a8c27bfd 	ldp	x29, x30, [sp], #32
    497c:	d65f03c0 	ret
		return NAN;
    4980:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4984:	9e670008 	fmov	d8, x0
    4988:	17fffff1 	b	494c <sptPrintAverageElapsedTime+0x38>
    498c:	d503201f 	nop

0000000000004990 <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    4990:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4994:	910003fd 	mov	x29, sp
    4998:	f9000bf3 	str	x19, [sp, #16]
	if(timer->use_cuda) {
    499c:	b9400013 	ldr	w19, [x0]
    49a0:	350000d3 	cbnz	w19, 49b8 <sptFreeTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	free(timer);
    49a4:	97fff287 	bl	13c0 <free@plt>
	return 0;
}
    49a8:	2a1303e0 	mov	w0, w19
    49ac:	f9400bf3 	ldr	x19, [sp, #16]
    49b0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    49b4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    49b8:	52800061 	mov	w1, #0x3                   	// #3
    49bc:	52800073 	mov	w19, #0x3                   	// #3
    49c0:	72a00041 	movk	w1, #0x2, lsl #16
    49c4:	52800b63 	mov	w3, #0x5b                  	// #91
    49c8:	72a00053 	movk	w19, #0x2, lsl #16
    49cc:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x120>
    49d0:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    49d4:	911e4084 	add	x4, x4, #0x790
    49d8:	911ee042 	add	x2, x2, #0x7b8
    49dc:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    49e0:	911f8000 	add	x0, x0, #0x7e0
    49e4:	9400006b 	bl	4b90 <spt_ComplainError>
}
    49e8:	2a1303e0 	mov	w0, w19
    49ec:	f9400bf3 	ldr	x19, [sp, #16]
    49f0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    49f4:	d65f03c0 	ret
    49f8:	d503201f 	nop
    49fc:	d503201f 	nop

0000000000004a00 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4a00:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
    4a04:	910003fd 	mov	x29, sp
    4a08:	a9025bf5 	stp	x21, x22, [sp, #32]
    4a0c:	aa0003f6 	mov	x22, x0
    4a10:	aa0103f5 	mov	x21, x1
    4a14:	a9046bf9 	stp	x25, x26, [sp, #64]
	int iores;
	sptIndex nrows = mtx->nrows;
	sptIndex ncols = mtx->ncols;
	sptIndex stride = mtx->stride;
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4a18:	aa0103e0 	mov	x0, x1
    4a1c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
	sptIndex ncols = mtx->ncols;
    4a20:	294066da 	ldp	w26, w25, [x22]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4a24:	91200021 	add	x1, x1, #0x800
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4a28:	a90153f3 	stp	x19, x20, [sp, #16]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4a2c:	2a1a03e2 	mov	w2, w26
    4a30:	2a1903e3 	mov	w3, w25
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4a34:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex stride = mtx->stride;
    4a38:	b9400edb 	ldr	w27, [x22, #12]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4a3c:	97fff289 	bl	1460 <fprintf@plt>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4a40:	37f80820 	tbnz	w0, #31, 4b44 <sptDumpMatrix+0x144>
	for(sptIndex i=0; i < nrows; ++i) {
    4a44:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex ncols = mtx->ncols;
    4a48:	2a1903f4 	mov	w20, w25
    4a4c:	b0000017 	adrp	x23, 5000 <_IO_stdin_used+0x120>
	for(sptIndex i=0; i < nrows; ++i) {
    4a50:	52800018 	mov	w24, #0x0                   	// #0
		for(sptIndex j=0; j < ncols; ++j) {
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4a54:	912122f7 	add	x23, x23, #0x848
	for(sptIndex i=0; i < nrows; ++i) {
    4a58:	3400061a 	cbz	w26, 4b18 <sptDumpMatrix+0x118>
    4a5c:	d503201f 	nop
		for(sptIndex j=0; j < ncols; ++j) {
    4a60:	4b190293 	sub	w19, w20, w25
    4a64:	35000099 	cbnz	w25, 4a74 <sptDumpMatrix+0x74>
    4a68:	14000025 	b	4afc <sptDumpMatrix+0xfc>
    4a6c:	6b13029f 	cmp	w20, w19
    4a70:	54000460 	b.eq	4afc <sptDumpMatrix+0xfc>  // b.none
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4a74:	f9400ac2 	ldr	x2, [x22, #16]
    4a78:	aa1703e1 	mov	x1, x23
    4a7c:	aa1503e0 	mov	x0, x21
    4a80:	bc735840 	ldr	s0, [x2, w19, uxtw #2]
		for(sptIndex j=0; j < ncols; ++j) {
    4a84:	11000673 	add	w19, w19, #0x1
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4a88:	1e22c000 	fcvt	d0, s0
    4a8c:	97fff275 	bl	1460 <fprintf@plt>
			spt_CheckOSError(iores < 0, "Mtx Dump");
    4a90:	36fffee0 	tbz	w0, #31, 4a6c <sptDumpMatrix+0x6c>
    4a94:	97fff267 	bl	1430 <__errno_location@plt>
    4a98:	b9400001 	ldr	w1, [x0]
    4a9c:	aa0003fc 	mov	x28, x0
    4aa0:	3140403f 	cmn	w1, #0x10, lsl #12
    4aa4:	54fffe40 	b.eq	4a6c <sptDumpMatrix+0x6c>  // b.none
    4aa8:	2a0103e0 	mov	w0, w1
    4aac:	b9006fe1 	str	w1, [sp, #108]
    4ab0:	97fff220 	bl	1330 <strerror@plt>
    4ab4:	aa0003e4 	mov	x4, x0
    4ab8:	b9406fe1 	ldr	w1, [sp, #108]
    4abc:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4ac0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    4ac4:	91204042 	add	x2, x2, #0x810
    4ac8:	9120e000 	add	x0, x0, #0x838
    4acc:	11404021 	add	w1, w1, #0x10, lsl #12
    4ad0:	528005c3 	mov	w3, #0x2e                  	// #46
    4ad4:	9400002f 	bl	4b90 <spt_ComplainError>
    4ad8:	a94363f7 	ldp	x23, x24, [sp, #48]
    4adc:	b9400380 	ldr	w0, [x28]
    4ae0:	11404000 	add	w0, w0, #0x10, lsl #12
		}
		iores = fprintf(fp, "\n");
	}
	iores = fprintf(fp, "\n");
	return 0;
}
    4ae4:	a94153f3 	ldp	x19, x20, [sp, #16]
    4ae8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4aec:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4af0:	a94573fb 	ldp	x27, x28, [sp, #80]
    4af4:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4af8:	d65f03c0 	ret
		iores = fprintf(fp, "\n");
    4afc:	aa1503e1 	mov	x1, x21
    4b00:	52800140 	mov	w0, #0xa                   	// #10
	for(sptIndex i=0; i < nrows; ++i) {
    4b04:	11000718 	add	w24, w24, #0x1
		iores = fprintf(fp, "\n");
    4b08:	97fff1be 	bl	1200 <fputc@plt>
	for(sptIndex i=0; i < nrows; ++i) {
    4b0c:	0b1b0294 	add	w20, w20, w27
    4b10:	6b18035f 	cmp	w26, w24
    4b14:	54fffa61 	b.ne	4a60 <sptDumpMatrix+0x60>  // b.any
	iores = fprintf(fp, "\n");
    4b18:	aa1503e1 	mov	x1, x21
    4b1c:	52800140 	mov	w0, #0xa                   	// #10
    4b20:	97fff1b8 	bl	1200 <fputc@plt>
	return 0;
    4b24:	52800000 	mov	w0, #0x0                   	// #0
}
    4b28:	a94153f3 	ldp	x19, x20, [sp, #16]
    4b2c:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    4b30:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    4b34:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4b38:	a94573fb 	ldp	x27, x28, [sp, #80]
    4b3c:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4b40:	d65f03c0 	ret
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4b44:	97fff23b 	bl	1430 <__errno_location@plt>
    4b48:	b9400014 	ldr	w20, [x0]
    4b4c:	aa0003f3 	mov	x19, x0
    4b50:	3140429f 	cmn	w20, #0x10, lsl #12
    4b54:	54fff780 	b.eq	4a44 <sptDumpMatrix+0x44>  // b.none
    4b58:	2a1403e0 	mov	w0, w20
    4b5c:	97fff1f5 	bl	1330 <strerror@plt>
    4b60:	11404281 	add	w1, w20, #0x10, lsl #12
    4b64:	aa0003e4 	mov	x4, x0
    4b68:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x120>
    4b6c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x120>
    4b70:	91204042 	add	x2, x2, #0x810
    4b74:	9120e000 	add	x0, x0, #0x838
    4b78:	52800543 	mov	w3, #0x2a                  	// #42
    4b7c:	94000005 	bl	4b90 <spt_ComplainError>
    4b80:	b9400260 	ldr	w0, [x19]
    4b84:	11404000 	add	w0, w0, #0x10, lsl #12
    4b88:	17ffffd7 	b	4ae4 <sptDumpMatrix+0xe4>
    4b8c:	d503201f 	nop

0000000000004b90 <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4b90:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4b94:	910003fd 	mov	x29, sp
    4b98:	a90153f3 	stp	x19, x20, [sp, #16]
	g_last_error.errcode = errcode;
    4b9c:	d53bd053 	mrs	x19, tpidr_el0
    4ba0:	91400273 	add	x19, x19, #0x0, lsl #12
    4ba4:	91004273 	add	x19, x19, #0x10
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4ba8:	a9025bf5 	stp	x21, x22, [sp, #32]
    4bac:	aa0003f4 	mov	x20, x0
    4bb0:	2a0103f5 	mov	w21, w1
    4bb4:	a90363f7 	stp	x23, x24, [sp, #48]
    4bb8:	aa0203f6 	mov	x22, x2
    4bbc:	2a0303f7 	mov	w23, w3
	g_last_error.module = module;
    4bc0:	f9000260 	str	x0, [x19]
	g_last_error.errcode = errcode;
    4bc4:	b9000a61 	str	w1, [x19, #8]
	g_last_error.file = file;
    4bc8:	f9000a62 	str	x2, [x19, #16]
	g_last_error.line = line;
    4bcc:	b9001a63 	str	w3, [x19, #24]
	if(reason) {
		free(g_last_error.reason);
    4bd0:	f9401266 	ldr	x6, [x19, #32]
	if(reason) {
    4bd4:	b4000324 	cbz	x4, 4c38 <spt_ComplainError+0xa8>
		free(g_last_error.reason);
    4bd8:	aa0603e0 	mov	x0, x6
    4bdc:	aa0403f8 	mov	x24, x4
    4be0:	97fff1f8 	bl	13c0 <free@plt>
		g_last_error.reason = strdup(reason);
    4be4:	aa1803e0 	mov	x0, x24
    4be8:	97fff1ca 	bl	1310 <strdup@plt>
    4bec:	f9001260 	str	x0, [x19, #32]
    4bf0:	aa0003e6 	mov	x6, x0
		if(!g_last_error.reason) {
    4bf4:	b4000440 	cbz	x0, 4c7c <spt_ComplainError+0xec>
			abort();
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4bf8:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4bfc:	394000c1 	ldrb	w1, [x6]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4c00:	f947d800 	ldr	x0, [x0, #4016]
    4c04:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4c08:	35000241 	cbnz	w1, 4c50 <spt_ComplainError+0xc0>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4c0c:	2a1703e5 	mov	w5, w23
    4c10:	aa1603e4 	mov	x4, x22
    4c14:	2a1503e3 	mov	w3, w21
    4c18:	aa1403e2 	mov	x2, x20
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    4c1c:	a94153f3 	ldp	x19, x20, [sp, #16]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4c20:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
}
    4c24:	a9425bf5 	ldp	x21, x22, [sp, #32]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4c28:	9121c021 	add	x1, x1, #0x870
}
    4c2c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4c30:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4c34:	17fff20b 	b	1460 <fprintf@plt>
    4c38:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    4c3c:	f947d800 	ldr	x0, [x0, #4016]
    4c40:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4c44:	b4fffe46 	cbz	x6, 4c0c <spt_ComplainError+0x7c>
    4c48:	394000c1 	ldrb	w1, [x6]
    4c4c:	34fffe01 	cbz	w1, 4c0c <spt_ComplainError+0x7c>
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4c50:	2a1703e5 	mov	w5, w23
    4c54:	aa1603e4 	mov	x4, x22
    4c58:	2a1503e3 	mov	w3, w21
    4c5c:	aa1403e2 	mov	x2, x20
}
    4c60:	a94153f3 	ldp	x19, x20, [sp, #16]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4c64:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
}
    4c68:	a9425bf5 	ldp	x21, x22, [sp, #32]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4c6c:	91214021 	add	x1, x1, #0x850
}
    4c70:	a94363f7 	ldp	x23, x24, [sp, #48]
    4c74:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4c78:	17fff1fa 	b	1460 <fprintf@plt>
			abort();
    4c7c:	97fff1bd 	bl	1370 <abort@plt>

0000000000004c80 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    4c80:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
	double size = (double)bytes;
	int suff = 0;
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
	while(size > 1024 && suff < 5) {
    4c84:	d2e81201 	mov	x1, #0x4090000000000000    	// #4652218415073722368
    4c88:	9e670021 	fmov	d1, x1
{
    4c8c:	910003fd 	mov	x29, sp
    4c90:	fd000be8 	str	d8, [sp, #16]
    4c94:	9e670008 	fmov	d8, x0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4c98:	f0000080 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    4c9c:	910a0000 	add	x0, x0, #0x280
	double size = (double)bytes;
    4ca0:	7e61d900 	ucvtf	d0, d8
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4ca4:	9100a3e1 	add	x1, sp, #0x28
    4ca8:	a9400c02 	ldp	x2, x3, [x0]
    4cac:	a9028fe2 	stp	x2, x3, [sp, #40]
	while(size > 1024 && suff < 5) {
    4cb0:	1e612010 	fcmpe	d0, d1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4cb4:	a9410c02 	ldp	x2, x3, [x0, #16]
    4cb8:	a9010c22 	stp	x2, x3, [x1, #16]
    4cbc:	f9401000 	ldr	x0, [x0, #32]
    4cc0:	f9001020 	str	x0, [x1, #32]
	while(size > 1024 && suff < 5) {
    4cc4:	54000669 	b.ls	4d90 <sptBytesString+0x110>  // b.plast
    4cc8:	d2e82600 	mov	x0, #0x4130000000000000    	// #4697254411347427328
    4ccc:	9e670002 	fmov	d2, x0
		size /= 1024.;
    4cd0:	d2e7ea00 	mov	x0, #0x3f50000000000000    	// #4562146422526312448
    4cd4:	9e670001 	fmov	d1, x0
	while(size > 1024 && suff < 5) {
    4cd8:	1e622010 	fcmpe	d0, d2
		size /= 1024.;
    4cdc:	1e610802 	fmul	d2, d0, d1
	while(size > 1024 && suff < 5) {
    4ce0:	540005c9 	b.ls	4d98 <sptBytesString+0x118>  // b.plast
    4ce4:	d2e83a00 	mov	x0, #0x41d0000000000000    	// #4742290407621132288
    4ce8:	9e670003 	fmov	d3, x0
		size /= 1024.;
    4cec:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4cf0:	1e632010 	fcmpe	d0, d3
    4cf4:	54000589 	b.ls	4da4 <sptBytesString+0x124>  // b.plast
    4cf8:	d2e84e00 	mov	x0, #0x4270000000000000    	// #4787326403894837248
    4cfc:	9e670003 	fmov	d3, x0
		size /= 1024.;
    4d00:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4d04:	1e632010 	fcmpe	d0, d3
    4d08:	540003e9 	b.ls	4d84 <sptBytesString+0x104>  // b.plast
    4d0c:	d2e86200 	mov	x0, #0x4310000000000000    	// #4832362400168542208
    4d10:	9e670003 	fmov	d3, x0
		size /= 1024.;
    4d14:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4d18:	1e632010 	fcmpe	d0, d3
    4d1c:	540004a9 	b.ls	4db0 <sptBytesString+0x130>  // b.plast
		size /= 1024.;
    4d20:	1e610840 	fmul	d0, d2, d1
		++suff;
    4d24:	528000a0 	mov	w0, #0x5                   	// #5
	}
	char * ret = NULL;
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4d28:	f860d822 	ldr	x2, [x1, w0, sxtw #3]
    4d2c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    4d30:	910083e0 	add	x0, sp, #0x20
    4d34:	9122e021 	add	x1, x1, #0x8b8
	char * ret = NULL;
    4d38:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4d3c:	97fff139 	bl	1220 <asprintf@plt>
    4d40:	3100041f 	cmn	w0, #0x1
    4d44:	540000a0 	b.eq	4d58 <sptBytesString+0xd8>  // b.none
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
		ret = NULL;
	}
	return ret;
}
    4d48:	fd400be8 	ldr	d8, [sp, #16]
	return ret;
    4d4c:	f94013e0 	ldr	x0, [sp, #32]
}
    4d50:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4d54:	d65f03c0 	ret
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    4d58:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf98c>
    4d5c:	9e660102 	fmov	x2, d8
    4d60:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x120>
    4d64:	91230021 	add	x1, x1, #0x8c0
    4d68:	f947d800 	ldr	x0, [x0, #4016]
    4d6c:	f9400000 	ldr	x0, [x0]
    4d70:	97fff1bc 	bl	1460 <fprintf@plt>
    4d74:	d2800000 	mov	x0, #0x0                   	// #0
}
    4d78:	fd400be8 	ldr	d8, [sp, #16]
    4d7c:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4d80:	d65f03c0 	ret
		size /= 1024.;
    4d84:	1e604040 	fmov	d0, d2
		++suff;
    4d88:	52800060 	mov	w0, #0x3                   	// #3
    4d8c:	17ffffe7 	b	4d28 <sptBytesString+0xa8>
	int suff = 0;
    4d90:	52800000 	mov	w0, #0x0                   	// #0
    4d94:	17ffffe5 	b	4d28 <sptBytesString+0xa8>
		size /= 1024.;
    4d98:	1e604040 	fmov	d0, d2
		++suff;
    4d9c:	52800020 	mov	w0, #0x1                   	// #1
    4da0:	17ffffe2 	b	4d28 <sptBytesString+0xa8>
		size /= 1024.;
    4da4:	1e604040 	fmov	d0, d2
		++suff;
    4da8:	52800040 	mov	w0, #0x2                   	// #2
    4dac:	17ffffdf 	b	4d28 <sptBytesString+0xa8>
		size /= 1024.;
    4db0:	1e604040 	fmov	d0, d2
		++suff;
    4db4:	52800080 	mov	w0, #0x4                   	// #4
    4db8:	17ffffdc 	b	4d28 <sptBytesString+0xa8>
    4dbc:	d503201f 	nop

0000000000004dc0 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    4dc0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4dc4:	910003fd 	mov	x29, sp
    4dc8:	fd000be8 	str	d8, [sp, #16]
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    4dcc:	97fff14d 	bl	1300 <rand@plt>
    4dd0:	1e220008 	scvtf	s8, w0
    4dd4:	52a61801 	mov	w1, #0x30c00000            	// #817889280
    4dd8:	1e270020 	fmov	s0, w1
    4ddc:	1e200908 	fmul	s8, s8, s0
	if(rand() % 2 == 0) {
    4de0:	97fff148 	bl	1300 <rand@plt>
		v *= -1;
    4de4:	1e214100 	fneg	s0, s8
    4de8:	f240001f 	tst	x0, #0x1
	}
	return v;
}
    4dec:	1e280c00 	fcsel	s0, s0, s8, eq  // eq = none
    4df0:	fd400be8 	ldr	d8, [sp, #16]
    4df4:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4df8:	d65f03c0 	ret
    4dfc:	d503201f 	nop

0000000000004e00 <__aarch64_cas4_acq_rel>:
    4e00:	d503245f 	bti	c
    4e04:	f0000090 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    4e08:	394aa610 	ldrb	w16, [x16, #681]
    4e0c:	34000070 	cbz	w16, 4e18 <__aarch64_cas4_acq_rel+0x18>
    4e10:	88e0fc41 	casal	w0, w1, [x2]
    4e14:	d65f03c0 	ret
    4e18:	2a0003f0 	mov	w16, w0
    4e1c:	885ffc40 	ldaxr	w0, [x2]
    4e20:	6b10001f 	cmp	w0, w16
    4e24:	54000061 	b.ne	4e30 <__aarch64_cas4_acq_rel+0x30>  // b.any
    4e28:	8811fc41 	stlxr	w17, w1, [x2]
    4e2c:	35ffff91 	cbnz	w17, 4e1c <__aarch64_cas4_acq_rel+0x1c>
    4e30:	d65f03c0 	ret
    4e34:	d503201f 	nop
    4e38:	d503201f 	nop
    4e3c:	d503201f 	nop

0000000000004e40 <__libc_csu_init>:
    4e40:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4e44:	910003fd 	mov	x29, sp
    4e48:	a90153f3 	stp	x19, x20, [sp, #16]
    4e4c:	d0000094 	adrp	x20, 16000 <__FRAME_END__+0xf98c>
    4e50:	9134e294 	add	x20, x20, #0xd38
    4e54:	a9025bf5 	stp	x21, x22, [sp, #32]
    4e58:	d0000095 	adrp	x21, 16000 <__FRAME_END__+0xf98c>
    4e5c:	913482b5 	add	x21, x21, #0xd20
    4e60:	cb150294 	sub	x20, x20, x21
    4e64:	2a0003f6 	mov	w22, w0
    4e68:	a90363f7 	stp	x23, x24, [sp, #48]
    4e6c:	aa0103f7 	mov	x23, x1
    4e70:	aa0203f8 	mov	x24, x2
    4e74:	9343fe94 	asr	x20, x20, #3
    4e78:	97fff0c6 	bl	1190 <_init>
    4e7c:	b4000174 	cbz	x20, 4ea8 <__libc_csu_init+0x68>
    4e80:	d2800013 	mov	x19, #0x0                   	// #0
    4e84:	d503201f 	nop
    4e88:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    4e8c:	aa1803e2 	mov	x2, x24
    4e90:	91000673 	add	x19, x19, #0x1
    4e94:	aa1703e1 	mov	x1, x23
    4e98:	2a1603e0 	mov	w0, w22
    4e9c:	d63f0060 	blr	x3
    4ea0:	eb13029f 	cmp	x20, x19
    4ea4:	54ffff21 	b.ne	4e88 <__libc_csu_init+0x48>  // b.any
    4ea8:	a94153f3 	ldp	x19, x20, [sp, #16]
    4eac:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4eb0:	a94363f7 	ldp	x23, x24, [sp, #48]
    4eb4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    4eb8:	d65f03c0 	ret
    4ebc:	d503201f 	nop

0000000000004ec0 <__libc_csu_fini>:
    4ec0:	d65f03c0 	ret

Disassembly of section .fini:

0000000000004ec4 <_fini>:
    4ec4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    4ec8:	910003fd 	mov	x29, sp
    4ecc:	a8c17bfd 	ldp	x29, x30, [sp], #16
    4ed0:	d65f03c0 	ret
