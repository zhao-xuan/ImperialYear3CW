
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
    11b4:	b00000b0 	adrp	x16, 16000 <__FRAME_END__+0xf89c>
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
    147c:	90000024 	adrp	x4, 5000 <_IO_stdin_used+0x30>
{
    1480:	a9007bfd 	stp	x29, x30, [sp]
    1484:	910003fd 	mov	x29, sp
    1488:	a90153f3 	stp	x19, x20, [sp, #16]
    148c:	2a0003f3 	mov	w19, w0
    1490:	aa0103f4 	mov	x20, x1
	printf("niters: %d\n", niters);
    1494:	9107c080 	add	x0, x4, #0x1f0
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
    14c0:	b00000ba 	adrp	x26, 16000 <__FRAME_END__+0xf89c>
    14c4:	d00000b7 	adrp	x23, 17000 <memcpy@GLIBC_2.17>
    14c8:	f947df5b 	ldr	x27, [x26, #4024]
    14cc:	9000003c 	adrp	x28, 5000 <_IO_stdin_used+0x30>
    14d0:	9123439c 	add	x28, x28, #0x8d0
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14d4:	910582f7 	add	x23, x23, #0x160
	bool random = true;
    14d8:	52800035 	mov	w21, #0x1                   	// #1
	FILE *fo = NULL;
    14dc:	d2800016 	mov	x22, #0x0                   	// #0
    14e0:	9101f3f9 	add	x25, sp, #0x7c
    14e4:	90000038 	adrp	x24, 5000 <_IO_stdin_used+0x30>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14e8:	aa1903e4 	mov	x4, x25
    14ec:	aa1703e3 	mov	x3, x23
		int option_index = 0;
    14f0:	b9007fff 	str	wzr, [sp, #124]
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14f4:	91080302 	add	x2, x24, #0x200
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
    152c:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    1530:	9101c3e2 	add	x2, sp, #0x70
    1534:	9108e021 	add	x1, x1, #0x238
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
    1550:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    1554:	9101d3e2 	add	x2, sp, #0x74
    1558:	9109a021 	add	x1, x1, #0x268
    155c:	f9400000 	ldr	x0, [x0]
    1560:	97ffffac 	bl	1410 <__isoc99_sscanf@plt>
				break;
    1564:	17ffffe1 	b	14e8 <main+0x78>
		switch(c) {
    1568:	7101901f 	cmp	w0, #0x64
    156c:	540034e1 	b.ne	1c08 <main+0x798>  // b.any
				sscanf(optarg, "%d", &dev_id);
    1570:	f947df40 	ldr	x0, [x26, #4024]
    1574:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    1578:	9101e3e2 	add	x2, sp, #0x78
    157c:	91090021 	add	x1, x1, #0x240
    1580:	f9400000 	ldr	x0, [x0]
    1584:	97ffffa3 	bl	1410 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    1588:	b9407be0 	ldr	w0, [sp, #120]
    158c:	11000800 	add	w0, w0, #0x2
    1590:	7100041f 	cmp	w0, #0x1
    1594:	54fffaa9 	b.ls	14e8 <main+0x78>  // b.plast
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    1598:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xf89c>
    159c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    15a0:	f947d863 	ldr	x3, [x3, #4016]
    15a4:	91092000 	add	x0, x0, #0x248
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
    15e4:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    15e8:	aa0203e1 	mov	x1, x2
    15ec:	9109c000 	add	x0, x0, #0x270
				printf("output file: %s\n", optarg); fflush(stdout);
    15f0:	97ffff8c 	bl	1420 <printf@plt>
    15f4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
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
    1620:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1624:	aa0203e1 	mov	x1, x2
    1628:	91084000 	add	x0, x0, #0x210
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
    1658:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    165c:	91088000 	add	x0, x0, #0x220
    1660:	17ffffe4 	b	15f0 <main+0x180>
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1664:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1668:	b94073e1 	ldr	w1, [sp, #112]
	printf("dev_id: %d\n", dev_id);

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    166c:	910223f9 	add	x25, sp, #0x88
	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1670:	910a4000 	add	x0, x0, #0x290
    1674:	97ffff6b 	bl	1420 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    1678:	b9407be1 	ldr	w1, [sp, #120]
    167c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1680:	910a8000 	add	x0, x0, #0x2a0
    1684:	97ffff67 	bl	1420 <printf@plt>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1688:	aa1903e0 	mov	x0, x25
    168c:	910323e2 	add	x2, sp, #0xc8
    1690:	52800021 	mov	w1, #0x1                   	// #1
    1694:	94000473 	bl	2860 <sptLoadSparseTensor>
    1698:	35001f60 	cbnz	w0, 1a84 <main+0x614>
	sptSparseTensorStatus(&X, stdout);
    169c:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xf89c>
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
    17e0:	94000bd8 	bl	4740 <sptNewTimer>
	sptStartTimer(timer);
    17e4:	f94043e0 	ldr	x0, [sp, #128]
    17e8:	528000b7 	mov	w23, #0x5                   	// #5
    17ec:	94000bf5 	bl	47c0 <sptStartTimer>

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
    1820:	94000c18 	bl	4880 <sptStopTimer>

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    1824:	f94043e0 	ldr	x0, [sp, #128]
    1828:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    182c:	910b0042 	add	x2, x2, #0x2c0
    1830:	528000a1 	mov	w1, #0x5                   	// #5
    1834:	94000c73 	bl	4a00 <sptPrintAverageElapsedTime>
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1838:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    183c:	b94077e5 	ldr	w5, [sp, #116]
	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    1840:	1e604001 	fmov	d1, d0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1844:	1e630262 	ucvtf	d2, w19
    1848:	fd420c03 	ldr	d3, [x0, #1048]
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
    19c0:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    19c4:	9e630002 	ucvtf	d2, x0
	printf("Performance: %.10lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    19c8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    19cc:	910e4000 	add	x0, x0, #0x390
	double gbw = (double)bytes / aver_time / 1e9;
    19d0:	fd420c23 	ldr	d3, [x1, #1048]
    19d4:	1e630842 	fmul	d2, d2, d3
	printf("Performance: %.10lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    19d8:	1e611841 	fdiv	d1, d2, d1
    19dc:	97fffe91 	bl	1420 <printf@plt>

	if(fo != NULL) {
    19e0:	b4000116 	cbz	x22, 1a00 <main+0x590>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    19e4:	f94037e0 	ldr	x0, [sp, #104]
    19e8:	aa1603e1 	mov	x1, x22
    19ec:	f8606a80 	ldr	x0, [x20, x0]
    19f0:	94000c40 	bl	4af0 <sptDumpMatrix>
    19f4:	35000480 	cbnz	w0, 1a84 <main+0x614>
		fclose(fo);
    19f8:	aa1603e0 	mov	x0, x22
    19fc:	97fffe15 	bl	1250 <fclose@plt>
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1a00:	d2800016 	mov	x22, #0x0                   	// #0
	}

	sptFreeTimer(timer);
    1a04:	f94043e0 	ldr	x0, [sp, #128]
    1a08:	94000c1e 	bl	4a80 <sptFreeTimer>
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
    1aa0:	94000aac 	bl	4550 <sptOmpMTTKRP>
    1aa4:	34ffeb80 	cbz	w0, 1814 <main+0x3a4>
    1aa8:	17fffff7 	b	1a84 <main+0x614>
		FILE* fPtr1 = fopen(fvname, "r");
    1aac:	90000033 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    1ab0:	910b6273 	add	x19, x19, #0x2d8
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
    1af8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1afc:	aa1603e2 	mov	x2, x22
    1b00:	aa1503e1 	mov	x1, x21
    1b04:	910ce000 	add	x0, x0, #0x338
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
    1b24:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1b28:	910b8000 	add	x0, x0, #0x2e0
    1b2c:	97fffe15 	bl	1380 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    1b30:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1b34:	910be000 	add	x0, x0, #0x2f8
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
    1b84:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1b88:	910ac000 	add	x0, x0, #0x2b0
    1b8c:	2a1c03e1 	mov	w1, w28
    1b90:	97fffe24 	bl	1420 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1b94:	b94073e3 	ldr	w3, [sp, #112]
    1b98:	2a1c03e4 	mov	w4, w28
    1b9c:	aa1803e2 	mov	x2, x24
    1ba0:	aa1403e1 	mov	x1, x20
    1ba4:	aa1903e0 	mov	x0, x25
    1ba8:	94000a6a 	bl	4550 <sptOmpMTTKRP>
    1bac:	34ffe160 	cbz	w0, 17d8 <main+0x368>
    1bb0:	17ffffb5 	b	1a84 <main+0x614>
	double gbw = (double)bytes / aver_time / 1e9;
    1bb4:	9e630002 	ucvtf	d2, x0
	printf("Performance: %.10lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1bb8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1bbc:	910e4000 	add	x0, x0, #0x390
	double gbw = (double)bytes / aver_time / 1e9;
    1bc0:	1e630842 	fmul	d2, d2, d3
	printf("Performance: %.10lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1bc4:	1e611841 	fdiv	d1, d2, d1
    1bc8:	97fffe16 	bl	1420 <printf@plt>
	if(fo != NULL) {
    1bcc:	b4000116 	cbz	x22, 1bec <main+0x77c>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1bd0:	f94037e0 	ldr	x0, [sp, #104]
    1bd4:	aa1603e1 	mov	x1, x22
    1bd8:	f8606a80 	ldr	x0, [x20, x0]
    1bdc:	94000bc5 	bl	4af0 <sptDumpMatrix>
    1be0:	35fff520 	cbnz	w0, 1a84 <main+0x614>
		fclose(fo);
    1be4:	aa1603e0 	mov	x0, x22
    1be8:	97fffd9a 	bl	1250 <fclose@plt>
	sptFreeTimer(timer);
    1bec:	f94043e0 	ldr	x0, [sp, #128]
    1bf0:	94000ba4 	bl	4a80 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1bf4:	17ffff8c 	b	1a24 <main+0x5b4>
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1bf8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1bfc:	910d8000 	add	x0, x0, #0x360
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
    1c74:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    1c78:	f947ec00 	ldr	x0, [x0, #4056]
    1c7c:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xf89c>
    1c80:	f947e863 	ldr	x3, [x3, #4048]
    1c84:	b00000a4 	adrp	x4, 16000 <__FRAME_END__+0xf89c>
    1c88:	f947cc84 	ldr	x4, [x4, #3992]
    1c8c:	97fffd85 	bl	12a0 <__libc_start_main@plt>
    1c90:	97fffdb8 	bl	1370 <abort@plt>

0000000000001c94 <call_weak_fn>:
    1c94:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
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
    1cc8:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xf89c>
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
    1d04:	b00000a2 	adrp	x2, 16000 <__FRAME_END__+0xf89c>
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
    1d38:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
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
    1db0:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP._omp_fn.0+0xb0>
static void print_usage(char ** argv) {
    1db4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
	printf("Usage: %s [options] \n\n", argv[0]);
    1db8:	aa0003e1 	mov	x1, x0
static void print_usage(char ** argv) {
    1dbc:	910003fd 	mov	x29, sp
	printf("Usage: %s [options] \n\n", argv[0]);
    1dc0:	913f6040 	add	x0, x2, #0xfd8
    1dc4:	97fffd97 	bl	1420 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1dc8:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0xb0>
    1dcc:	913fc000 	add	x0, x0, #0xff0
    1dd0:	97fffd6c 	bl	1380 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    1dd4:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1dd8:	91008000 	add	x0, x0, #0x20
    1ddc:	97fffd69 	bl	1380 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1de0:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1de4:	91016000 	add	x0, x0, #0x58
    1de8:	97fffd66 	bl	1380 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    1dec:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1df0:	91030000 	add	x0, x0, #0xc0
    1df4:	97fffd63 	bl	1380 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    1df8:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1dfc:	91044000 	add	x0, x0, #0x110
    1e00:	97fffd60 	bl	1380 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    1e04:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1e08:	91054000 	add	x0, x0, #0x150
    1e0c:	97fffd5d 	bl	1380 <puts@plt>
	printf("         --help\n");
    1e10:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1e14:	91078000 	add	x0, x0, #0x1e0
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
    1efc:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    1f00:	4f000482 	movi	v2.4s, #0x4
    1f04:	aa0003e3 	mov	x3, x0
    1f08:	53027e64 	lsr	w4, w19, #2
    1f0c:	8b245004 	add	x4, x0, w4, uxtw #4
    1f10:	3dc11820 	ldr	q0, [x1, #1120]
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
    1fa8:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    1fac:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    1fb0:	91108042 	add	x2, x2, #0x420
    1fb4:	91112000 	add	x0, x0, #0x448
    1fb8:	d2800004 	mov	x4, #0x0                   	// #0
    1fbc:	52800643 	mov	w3, #0x32                  	// #50
    1fc0:	2a1303e1 	mov	w1, w19
    1fc4:	94000b2f 	bl	4c80 <spt_ComplainError>
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
    2020:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    2024:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    2028:	91108042 	add	x2, x2, #0x420
    202c:	91112000 	add	x0, x0, #0x448
    2030:	d2800004 	mov	x4, #0x0                   	// #0
    2034:	528006a3 	mov	w3, #0x35                  	// #53
    2038:	2a1303e1 	mov	w1, w19
    203c:	94000b11 	bl	4c80 <spt_ComplainError>
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
    2140:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    2144:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    2148:	aa0003e4 	mov	x4, x0
    214c:	11404281 	add	w1, w20, #0x10, lsl #12
    2150:	91126060 	add	x0, x3, #0x498
    2154:	9111c042 	add	x2, x2, #0x470
    2158:	528005c3 	mov	w3, #0x2e                  	// #46
    215c:	94000ac9 	bl	4c80 <spt_ComplainError>
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
    231c:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    2320:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    2324:	aa0003e4 	mov	x4, x0
    2328:	114042e1 	add	w1, w23, #0x10, lsl #12
    232c:	9112a060 	add	x0, x3, #0x4a8
    2330:	9111c042 	add	x2, x2, #0x470
    2334:	52800a43 	mov	w3, #0x52                  	// #82
    2338:	94000a52 	bl	4c80 <spt_ComplainError>
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
    23f0:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    23f4:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    23f8:	aa0003e4 	mov	x4, x0
    23fc:	11404301 	add	w1, w24, #0x10, lsl #12
    2400:	9112e060 	add	x0, x3, #0x4b8
    2404:	9111c042 	add	x2, x2, #0x470
    2408:	52800d23 	mov	w3, #0x69                  	// #105
    240c:	94000a1d 	bl	4c80 <spt_ComplainError>
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
    24a0:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    24a4:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    24a8:	aa0003e4 	mov	x4, x0
    24ac:	11404281 	add	w1, w20, #0x10, lsl #12
    24b0:	91132060 	add	x0, x3, #0x4c8
    24b4:	9111c042 	add	x2, x2, #0x470
    24b8:	52801283 	mov	w3, #0x94                  	// #148
    24bc:	940009f1 	bl	4c80 <spt_ComplainError>
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
    2564:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    2568:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    256c:	aa0003e4 	mov	x4, x0
    2570:	11404301 	add	w1, w24, #0x10, lsl #12
    2574:	91136060 	add	x0, x3, #0x4d8
    2578:	9111c042 	add	x2, x2, #0x470
    257c:	52801543 	mov	w3, #0xaa                  	// #170
    2580:	940009c0 	bl	4c80 <spt_ComplainError>
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
    2640:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    2644:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    2648:	aa0003e4 	mov	x4, x0
    264c:	11404301 	add	w1, w24, #0x10, lsl #12
    2650:	9113a060 	add	x0, x3, #0x4e8
    2654:	9111c042 	add	x2, x2, #0x470
    2658:	52801823 	mov	w3, #0xc1                  	// #193
    265c:	94000989 	bl	4c80 <spt_ComplainError>
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
    287c:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    2880:	910b6021 	add	x1, x1, #0x2d8
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
    28ac:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    28b0:	9113e021 	add	x1, x1, #0x4f8
    28b4:	97fffabb 	bl	13a0 <strcmp@plt>
    28b8:	34000a00 	cbz	w0, 29f8 <sptLoadSparseTensor+0x198>
    28bc:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    28c0:	91140021 	add	x1, x1, #0x500
    28c4:	aa1603e0 	mov	x0, x22
    28c8:	97fffab6 	bl	13a0 <strcmp@plt>
    28cc:	34003f00 	cbz	w0, 30ac <sptLoadSparseTensor+0x84c>
    28d0:	aa1603e0 	mov	x0, x22
    28d4:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    28d8:	91142021 	add	x1, x1, #0x508
    28dc:	97fffab1 	bl	13a0 <strcmp@plt>
	} while(file_extensions[++idx].extension != NULL);
    28e0:	d2800041 	mov	x1, #0x2                   	// #2
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    28e4:	340008c0 	cbz	w0, 29fc <sptLoadSparseTensor+0x19c>
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    28e8:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    28ec:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    28f0:	a9046bf9 	stp	x25, x26, [sp, #64]
    28f4:	aa1503e2 	mov	x2, x21
    28f8:	91144021 	add	x1, x1, #0x510
    28fc:	f947d800 	ldr	x0, [x0, #4016]
    2900:	f9400000 	ldr	x0, [x0]
    2904:	97fffad7 	bl	1460 <fprintf@plt>
	iores = fscanf(fp, "%u", &tsr->nmodes);
    2908:	f0000018 	adrp	x24, 5000 <_IO_stdin_used+0x30>
    290c:	aa1303e2 	mov	x2, x19
    2910:	9108e301 	add	x1, x24, #0x238
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
    2954:	9108e319 	add	x25, x24, #0x238
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
    29a4:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    29a8:	97fffa62 	bl	1330 <strerror@plt>
    29ac:	f0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    29b0:	aa0003e4 	mov	x4, x0
    29b4:	11404341 	add	w1, w26, #0x10, lsl #12
    29b8:	911642a0 	add	x0, x21, #0x590
    29bc:	9115a262 	add	x2, x19, #0x568
    29c0:	52800a63 	mov	w3, #0x53                  	// #83
    29c4:	940008af 	bl	4c80 <spt_ComplainError>
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
    29fc:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
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
    2c10:	f0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    2c14:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    2c18:	2a0003e1 	mov	w1, w0
    2c1c:	9115a262 	add	x2, x19, #0x568
    2c20:	911a2060 	add	x0, x3, #0x688
    2c24:	d2800004 	mov	x4, #0x0                   	// #0
    2c28:	528020a3 	mov	w3, #0x105                 	// #261
    2c2c:	94000815 	bl	4c80 <spt_ComplainError>
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
    2ca4:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    2ca8:	f0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    2cac:	2a0003e1 	mov	w1, w0
    2cb0:	9115a262 	add	x2, x19, #0x568
    2cb4:	911642a0 	add	x0, x21, #0x590
    2cb8:	d2800004 	mov	x4, #0x0                   	// #0
    2cbc:	52800b43 	mov	w3, #0x5a                  	// #90
    2cc0:	940007f0 	bl	4c80 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2cc4:	97fff9db 	bl	1430 <__errno_location@plt>
    2cc8:	b9400017 	ldr	w23, [x0]
    2ccc:	aa0003f6 	mov	x22, x0
    2cd0:	314042ff 	cmn	w23, #0x10, lsl #12
    2cd4:	54ffe800 	b.eq	29d4 <sptLoadSparseTensor+0x174>  // b.none
    2cd8:	2a1703e0 	mov	w0, w23
    2cdc:	97fff995 	bl	1330 <strerror@plt>
    2ce0:	aa0003e4 	mov	x4, x0
    2ce4:	9115a262 	add	x2, x19, #0x568
    2ce8:	911642a0 	add	x0, x21, #0x590
    2cec:	114042e1 	add	w1, w23, #0x10, lsl #12
    2cf0:	52802483 	mov	w3, #0x124                 	// #292
    2cf4:	940007e3 	bl	4c80 <spt_ComplainError>
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
    2ed4:	900000a1 	adrp	x1, 16000 <__FRAME_END__+0xf89c>
    2ed8:	d37df003 	lsl	x3, x0, #3
    2edc:	f947d820 	ldr	x0, [x1, #4016]
    2ee0:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    2ee4:	aa0303e2 	mov	x2, x3
    2ee8:	91184021 	add	x1, x1, #0x610
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
    2f1c:	9108e318 	add	x24, x24, #0x238
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2f20:	52800316 	mov	w22, #0x18                  	// #24
    2f24:	a90573fb 	stp	x27, x28, [sp, #80]
			iores = fscanf(fp, "%lf", &value);
    2f28:	f000001b 	adrp	x27, 5000 <_IO_stdin_used+0x30>
    2f2c:	9116e37b 	add	x27, x27, #0x5b8
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
    2ff4:	f0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    2ff8:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    2ffc:	f0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    3000:	91168084 	add	x4, x4, #0x5a0
    3004:	9115a262 	add	x2, x19, #0x568
    3008:	911642a0 	add	x0, x21, #0x590
    300c:	52800d03 	mov	w3, #0x68                  	// #104
    3010:	52800061 	mov	w1, #0x3                   	// #3
    3014:	9400071b 	bl	4c80 <spt_ComplainError>
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
    3038:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    303c:	97fff8bd 	bl	1330 <strerror@plt>
    3040:	d0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    3044:	aa0003e4 	mov	x4, x0
    3048:	11404321 	add	w1, w25, #0x10, lsl #12
    304c:	911642a0 	add	x0, x21, #0x590
    3050:	9115a262 	add	x2, x19, #0x568
    3054:	52800943 	mov	w3, #0x4a                  	// #74
    3058:	9400070a 	bl	4c80 <spt_ComplainError>
    305c:	b94002c0 	ldr	w0, [x22]
    3060:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    3064:	34ffcb80 	cbz	w0, 29d4 <sptLoadSparseTensor+0x174>
    3068:	17ffff17 	b	2cc4 <sptLoadSparseTensor+0x464>
			spt_CheckOSError(iores != 0, "SpTns Load");
    306c:	2a1603e0 	mov	w0, w22
    3070:	97fff8b0 	bl	1330 <strerror@plt>
    3074:	d0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    3078:	aa0003e4 	mov	x4, x0
    307c:	9115a262 	add	x2, x19, #0x568
    3080:	91164060 	add	x0, x3, #0x590
    3084:	114042c1 	add	w1, w22, #0x10, lsl #12
    3088:	52802503 	mov	w3, #0x128                 	// #296
    308c:	940006fd 	bl	4c80 <spt_ComplainError>
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
    30f0:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    30f4:	97fff88f 	bl	1330 <strerror@plt>
    30f8:	d0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    30fc:	aa0003e4 	mov	x4, x0
    3100:	11404341 	add	w1, w26, #0x10, lsl #12
    3104:	911642a0 	add	x0, x21, #0x590
    3108:	9115a262 	add	x2, x19, #0x568
    310c:	52800a03 	mov	w3, #0x50                  	// #80
    3110:	940006dc 	bl	4c80 <spt_ComplainError>
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
    313c:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    3140:	97fff87c 	bl	1330 <strerror@plt>
    3144:	d0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    3148:	aa0003e4 	mov	x4, x0
    314c:	11404341 	add	w1, w26, #0x10, lsl #12
    3150:	911642a0 	add	x0, x21, #0x590
    3154:	9115a262 	add	x2, x19, #0x568
    3158:	528009a3 	mov	w3, #0x4d                  	// #77
    315c:	940006c9 	bl	4c80 <spt_ComplainError>
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
    31c0:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    31c4:	97fff85b 	bl	1330 <strerror@plt>
    31c8:	d0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    31cc:	aa0003e4 	mov	x4, x0
    31d0:	11404361 	add	w1, w27, #0x10, lsl #12
    31d4:	911642a0 	add	x0, x21, #0x590
    31d8:	9115a262 	add	x2, x19, #0x568
    31dc:	52800ae3 	mov	w3, #0x57                  	// #87
    31e0:	940006a8 	bl	4c80 <spt_ComplainError>
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
    3200:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    3204:	d0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    3208:	2a0003e1 	mov	w1, w0
    320c:	9115a262 	add	x2, x19, #0x568
    3210:	911a2060 	add	x0, x3, #0x688
    3214:	d2800004 	mov	x4, #0x0                   	// #0
    3218:	52802103 	mov	w3, #0x108                 	// #264
    321c:	94000699 	bl	4c80 <spt_ComplainError>
    3220:	17fffe84 	b	2c30 <sptLoadSparseTensor+0x3d0>
	spt_CheckError(retval, "SpTns Load", NULL);
    3224:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x30>
    3228:	d0000015 	adrp	x21, 5000 <_IO_stdin_used+0x30>
    322c:	2a0003e1 	mov	w1, w0
    3230:	9115a262 	add	x2, x19, #0x568
    3234:	911642a0 	add	x0, x21, #0x590
    3238:	d2800004 	mov	x4, #0x0                   	// #0
    323c:	52800ba3 	mov	w3, #0x5d                  	// #93
    3240:	94000690 	bl	4c80 <spt_ComplainError>
    3244:	17fffea0 	b	2cc4 <sptLoadSparseTensor+0x464>
	sptAssert(fp != NULL);
    3248:	12800000 	mov	w0, #0xffffffff            	// #-1
    324c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3250:	a90573fb 	stp	x27, x28, [sp, #80]
    3254:	97fff7e3 	bl	11e0 <exit@plt>
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    3258:	f0000080 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    325c:	d37df023 	lsl	x3, x1, #3
    3260:	f947d800 	ldr	x0, [x0, #4016]
    3264:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3268:	aa0303e2 	mov	x2, x3
    326c:	91170021 	add	x1, x1, #0x5c0
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
    3310:	d0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    3314:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    3318:	aa0003e4 	mov	x4, x0
    331c:	114042c1 	add	w1, w22, #0x10, lsl #12
    3320:	911b0060 	add	x0, x3, #0x6c0
    3324:	911a6042 	add	x2, x2, #0x698
    3328:	52800703 	mov	w3, #0x38                  	// #56
    332c:	94000655 	bl	4c80 <spt_ComplainError>
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
    33a0:	940006c4 	bl	4eb0 <sptRandomValue>
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
    3424:	940006a3 	bl	4eb0 <sptRandomValue>
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
    36f4:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    36f8:	aa1503e3 	mov	x3, x21
    36fc:	d0000016 	adrp	x22, 5000 <_IO_stdin_used+0x30>
    3700:	911cc2d6 	add	x22, x22, #0x730
{
    3704:	a90153f3 	stp	x19, x20, [sp, #16]
    3708:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    370c:	911b2020 	add	x0, x1, #0x6c8
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
    3724:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3728:	aa1503e0 	mov	x0, x21
    372c:	911c4021 	add	x1, x1, #0x710
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
    3770:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3774:	aa1503e0 	mov	x0, x21
    3778:	911c8021 	add	x1, x1, #0x720
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
    3944:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3948:	aa1503e0 	mov	x0, x21
    394c:	911ce021 	add	x1, x1, #0x738
    3950:	97fff6c4 	bl	1460 <fprintf@plt>

	fprintf(fp, "Average slice length (c): ");
    3954:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3958:	aa1503e3 	mov	x3, x21
    395c:	911d2000 	add	x0, x0, #0x748
    3960:	d2800342 	mov	x2, #0x1a                  	// #26
    3964:	d2800021 	mov	x1, #0x1                   	// #1
    3968:	97fff69a 	bl	13d0 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    396c:	b9400260 	ldr	w0, [x19]
    3970:	7100041f 	cmp	w0, #0x1
    3974:	54000640 	b.eq	3a3c <sptSparseTensorStatus+0x35c>  // b.none
    3978:	d0000016 	adrp	x22, 5000 <_IO_stdin_used+0x30>
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    397c:	911da2d6 	add	x22, x22, #0x768
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
    39c8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    39cc:	aa1503e0 	mov	x0, x21
    39d0:	911de021 	add	x1, x1, #0x778
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
    39fc:	940004dd 	bl	4d70 <sptBytesString>
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3a00:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3a04:	aa0003e2 	mov	x2, x0
	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    3a08:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3a0c:	911e0021 	add	x1, x1, #0x780
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
    3b58:	940002fa 	bl	4740 <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    3b5c:	f94047e0 	ldr	x0, [sp, #136]
    3b60:	94000318 	bl	47c0 <sptStartTimer>
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
    3bf0:	94000324 	bl	4880 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    3bf4:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3bf8:	911fa021 	add	x1, x1, #0x7e8
    3bfc:	f94047e0 	ldr	x0, [sp, #136]
    3c00:	94000360 	bl	4980 <sptPrintElapsedTime>
    3c04:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3c08:	f94047e0 	ldr	x0, [sp, #136]
    3c0c:	9400039d 	bl	4a80 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3c10:	1e604100 	fmov	d0, d8
    3c14:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3c18:	91208000 	add	x0, x0, #0x820
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
    3c4c:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    3c50:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    3c54:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3c58:	911e6084 	add	x4, x4, #0x798
    3c5c:	911fa000 	add	x0, x0, #0x7e8
    3c60:	911f0042 	add	x2, x2, #0x7c0
    3c64:	52801403 	mov	w3, #0xa0                  	// #160
    3c68:	52800041 	mov	w1, #0x2                   	// #2
    3c6c:	94000405 	bl	4c80 <spt_ComplainError>
    3c70:	52800040 	mov	w0, #0x2                   	// #2
    3c74:	17ffffef 	b	3c30 <sptMTTKRP_3D+0x1e0>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3c78:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    3c7c:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    3c80:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3c84:	91200084 	add	x4, x4, #0x800
    3c88:	911fa000 	add	x0, x0, #0x7e8
    3c8c:	911f0042 	add	x2, x2, #0x7c0
    3c90:	52801463 	mov	w3, #0xa3                  	// #163
    3c94:	52800041 	mov	w1, #0x2                   	// #2
    3c98:	940003fa 	bl	4c80 <spt_ComplainError>
    3c9c:	52800040 	mov	w0, #0x2                   	// #2
    3ca0:	17ffffe4 	b	3c30 <sptMTTKRP_3D+0x1e0>
	sptAssert(nmodes ==3);
    3ca4:	12800000 	mov	w0, #0xffffffff            	// #-1
    3ca8:	fd0033e8 	str	d8, [sp, #96]
    3cac:	97fff54d 	bl	11e0 <exit@plt>

0000000000003cb0 <sptMTTKRP>:
							sptIndex const mode) {
    3cb0:	a9b77bfd 	stp	x29, x30, [sp, #-144]!
    3cb4:	910003fd 	mov	x29, sp
	sptIndex const nmodes = X->nmodes;
    3cb8:	b9400007 	ldr	w7, [x0]
	if(nmodes == 3) {
    3cbc:	71000cff 	cmp	w7, #0x3
    3cc0:	54001320 	b.eq	3f24 <sptMTTKRP+0x274>  // b.none
	sptNnzIndex const nnz = X->nnz;
    3cc4:	a9025bf5 	stp	x21, x22, [sp, #32]
    3cc8:	2a0303e9 	mov	w9, w3
    3ccc:	aa0103f6 	mov	x22, x1
    3cd0:	a90573fb 	stp	x27, x28, [sp, #80]
    3cd4:	aa0003fc 	mov	x28, x0
	sptIndex const * const ndims = X->ndims;
    3cd8:	aa0203fb 	mov	x27, x2
	sptMatrix * const restrict M = mats[nmodes];
    3cdc:	f8675820 	ldr	x0, [x1, w7, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    3ce0:	f9400b88 	ldr	x8, [x28, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    3ce4:	340003a7 	cbz	w7, 3d58 <sptMTTKRP+0xa8>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3ce8:	d2800004 	mov	x4, #0x0                   	// #0
    3cec:	b9400401 	ldr	w1, [x0, #4]
    3cf0:	14000008 	b	3d10 <sptMTTKRP+0x60>
		if(mats[i]->nrows != ndims[i]) {
    3cf4:	b94000a6 	ldr	w6, [x5]
    3cf8:	b8647905 	ldr	w5, [x8, x4, lsl #2]
    3cfc:	6b0500df 	cmp	w6, w5
    3d00:	54001021 	b.ne	3f04 <sptMTTKRP+0x254>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3d04:	91000484 	add	x4, x4, #0x1
    3d08:	6b0400ff 	cmp	w7, w4
    3d0c:	54000269 	b.ls	3d58 <sptMTTKRP+0xa8>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3d10:	f8647ac5 	ldr	x5, [x22, x4, lsl #3]
    3d14:	b94004a6 	ldr	w6, [x5, #4]
    3d18:	6b0100df 	cmp	w6, w1
    3d1c:	54fffec0 	b.eq	3cf4 <sptMTTKRP+0x44>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3d20:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    3d24:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    3d28:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3d2c:	911e6084 	add	x4, x4, #0x798
    3d30:	911f0042 	add	x2, x2, #0x7c0
    3d34:	911fa000 	add	x0, x0, #0x7e8
    3d38:	528007c3 	mov	w3, #0x3e                  	// #62
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3d3c:	52800041 	mov	w1, #0x2                   	// #2
    3d40:	940003d0 	bl	4c80 <spt_ComplainError>
    3d44:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3d48:	52800040 	mov	w0, #0x2                   	// #2
    3d4c:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3d50:	a8c97bfd 	ldp	x29, x30, [sp], #144
    3d54:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    3d58:	f8695ac5 	ldr	x5, [x22, w9, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d5c:	52800304 	mov	w4, #0x18                  	// #24
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d60:	52800001 	mov	w1, #0x0                   	// #0
    3d64:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex const stride = mats[0]->stride;
    3d68:	f94002c2 	ldr	x2, [x22]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d6c:	a9418f98 	ldp	x24, x3, [x28, #24]
    3d70:	a90153f3 	stp	x19, x20, [sp, #16]
	sptNewValueVector(&scratch, R, R);
    3d74:	9101e3f4 	add	x20, sp, #0x78
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d78:	9ba40d23 	umaddl	x3, w9, w4, x3
	sptIndex const stride = mats[0]->stride;
    3d7c:	b9400c57 	ldr	w23, [x2, #12]
	sptIndex const R = mats[mode]->ncols;
    3d80:	a9046bf9 	stp	x25, x26, [sp, #64]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d84:	b94000a2 	ldr	w2, [x5]
	sptValue * const restrict mvals = M->values;
    3d88:	f9400813 	ldr	x19, [x0, #16]
	sptIndex const R = mats[mode]->ncols;
    3d8c:	b94004b5 	ldr	w21, [x5, #4]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d90:	1b027ee2 	mul	w2, w23, w2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d94:	f940087a 	ldr	x26, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d98:	aa1303e0 	mov	x0, x19
    3d9c:	d37ef442 	lsl	x2, x2, #2
	sptValue const * const restrict vals = X->values.data;
    3da0:	f9401f99 	ldr	x25, [x28, #56]
    3da4:	fd0033e8 	str	d8, [sp, #96]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3da8:	97fff54a 	bl	12d0 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    3dac:	2a1503e2 	mov	w2, w21
    3db0:	aa1403e0 	mov	x0, x20
    3db4:	aa0203e1 	mov	x1, x2
    3db8:	97fff8c6 	bl	20d0 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    3dbc:	0f000400 	movi	v0.2s, #0x0
    3dc0:	aa1403e0 	mov	x0, x20
    3dc4:	97fff8eb 	bl	2170 <sptConstantValueVector>
	sptNewTimer(&timer, 0);
    3dc8:	52800001 	mov	w1, #0x0                   	// #0
    3dcc:	9101c3e0 	add	x0, sp, #0x70
    3dd0:	9400025c 	bl	4740 <sptNewTimer>
	sptStartTimer(timer);
    3dd4:	f9403be0 	ldr	x0, [sp, #112]
    3dd8:	9400027a 	bl	47c0 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3ddc:	52800304 	mov	w4, #0x18                  	// #24
    3de0:	b40005f8 	cbz	x24, 3e9c <sptMTTKRP+0x1ec>
		sptIndex times_mat_index_2 = mats_order[2];
    3de4:	29408762 	ldp	w2, w1, [x27, #4]
		sptIndex times_mat_index_3 = mats_order[3];
    3de8:	b9400f60 	ldr	w0, [x27, #12]
		sptIndex * times_inds_1 = X->inds[times_mat_index_1].data;
    3dec:	f9401383 	ldr	x3, [x28, #32]
		sptValue* times_mat_values_1 = times_mat_1->values + tmp_mult_1;
    3df0:	f8625ac7 	ldr	x7, [x22, w2, uxtw #3]
		sptValue* times_mat_values_2 = times_mat_2->values + tmp_mult_2;
    3df4:	f8615ac6 	ldr	x6, [x22, w1, uxtw #3]
		sptIndex * times_inds_1 = X->inds[times_mat_index_1].data;
    3df8:	9ba40c42 	umaddl	x2, w2, w4, x3
		sptValue* times_mat_values_3 = times_mat_3->values + tmp_mult_3;
    3dfc:	f8605ac5 	ldr	x5, [x22, w0, uxtw #3]
		sptIndex * times_inds_2 = X->inds[times_mat_index_2].data;
    3e00:	9ba40c21 	umaddl	x1, w1, w4, x3
		sptIndex * times_inds_3 = X->inds[times_mat_index_3].data;
    3e04:	9ba40c00 	umaddl	x0, w0, w4, x3
		sptValue* times_mat_values_1 = times_mat_1->values + tmp_mult_1;
    3e08:	f94008e7 	ldr	x7, [x7, #16]
		sptValue* times_mat_values_2 = times_mat_2->values + tmp_mult_2;
    3e0c:	f94008c8 	ldr	x8, [x6, #16]
		sptValue* times_mat_values_3 = times_mat_3->values + tmp_mult_3;
    3e10:	f94008a9 	ldr	x9, [x5, #16]
		sptIndex * times_inds_3 = X->inds[times_mat_index_3].data;
    3e14:	f940080c 	ldr	x12, [x0, #16]
		sptIndex * times_inds_2 = X->inds[times_mat_index_2].data;
    3e18:	f940082b 	ldr	x11, [x1, #16]
		sptIndex * times_inds_1 = X->inds[times_mat_index_1].data;
    3e1c:	f940084a 	ldr	x10, [x2, #16]
    3e20:	340003f5 	cbz	w21, 3e9c <sptMTTKRP+0x1ec>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3e24:	d2800006 	mov	x6, #0x0                   	// #0
		sptIndex tmp_mult_1 = tmp_i_1 * stride;
    3e28:	b8667944 	ldr	w4, [x10, x6, lsl #2]
		sptIndex tmp_mode = mode_i * stride;
    3e2c:	d2800000 	mov	x0, #0x0                   	// #0
		sptIndex tmp_mult_2 = tmp_i_2 * stride;
    3e30:	b8667963 	ldr	w3, [x11, x6, lsl #2]
		sptIndex tmp_mult_3 = tmp_i_3 * stride;
    3e34:	b8667982 	ldr	w2, [x12, x6, lsl #2]
    3e38:	1b047ee4 	mul	w4, w23, w4
		sptIndex tmp_mode = mode_i * stride;
    3e3c:	b8667b45 	ldr	w5, [x26, x6, lsl #2]
    3e40:	1b037ee3 	mul	w3, w23, w3
		sptValue const entry = vals[x];
    3e44:	bc667b24 	ldr	s4, [x25, x6, lsl #2]
		sptIndex tmp_mode = mode_i * stride;
    3e48:	8b0408e4 	add	x4, x7, x4, lsl #2
    3e4c:	1b027ee2 	mul	w2, w23, w2
    3e50:	8b030903 	add	x3, x8, x3, lsl #2
    3e54:	1b057ee5 	mul	w5, w23, w5
		for(sptIndex r=0; r<R; ++r) {
    3e58:	8b020922 	add	x2, x9, x2, lsl #2
    3e5c:	d503201f 	nop
			mvals[tmp_mode + r] += tmp;
    3e60:	bc607842 	ldr	s2, [x2, x0, lsl #2]
    3e64:	0b0000a1 	add	w1, w5, w0
    3e68:	bc607880 	ldr	s0, [x4, x0, lsl #2]
    3e6c:	bc607863 	ldr	s3, [x3, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3e70:	91000400 	add	x0, x0, #0x1
    3e74:	6b0002bf 	cmp	w21, w0
			mvals[tmp_mode + r] += tmp;
    3e78:	bc617a61 	ldr	s1, [x19, x1, lsl #2]
    3e7c:	1e220882 	fmul	s2, s4, s2
    3e80:	1e230800 	fmul	s0, s0, s3
    3e84:	1f000440 	fmadd	s0, s2, s0, s1
    3e88:	bc217a60 	str	s0, [x19, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3e8c:	54fffea8 	b.hi	3e60 <sptMTTKRP+0x1b0>  // b.pmore
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3e90:	910004c6 	add	x6, x6, #0x1
    3e94:	eb06031f 	cmp	x24, x6
    3e98:	54fffc81 	b.ne	3e28 <sptMTTKRP+0x178>  // b.any
	sptStopTimer(timer);
    3e9c:	f9403be0 	ldr	x0, [sp, #112]
    3ea0:	94000278 	bl	4880 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    3ea4:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    3ea8:	911fa021 	add	x1, x1, #0x7e8
    3eac:	f9403be0 	ldr	x0, [sp, #112]
    3eb0:	940002b4 	bl	4980 <sptPrintElapsedTime>
    3eb4:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3eb8:	f9403be0 	ldr	x0, [sp, #112]
    3ebc:	940002f1 	bl	4a80 <sptFreeTimer>
	sptFreeValueVector(&scratch);
    3ec0:	aa1403e0 	mov	x0, x20
    3ec4:	97fff957 	bl	2420 <sptFreeValueVector>
	printf("[Total time]: %lf\n", total_time);
    3ec8:	1e604100 	fmov	d0, d8
    3ecc:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3ed0:	91208000 	add	x0, x0, #0x820
    3ed4:	97fff553 	bl	1420 <printf@plt>
	printf("\n");
    3ed8:	52800140 	mov	w0, #0xa                   	// #10
    3edc:	97fff55d 	bl	1450 <putchar@plt>
	return 0;
    3ee0:	52800000 	mov	w0, #0x0                   	// #0
    3ee4:	a94153f3 	ldp	x19, x20, [sp, #16]
    3ee8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3eec:	a94363f7 	ldp	x23, x24, [sp, #48]
    3ef0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3ef4:	a94573fb 	ldp	x27, x28, [sp, #80]
    3ef8:	fd4033e8 	ldr	d8, [sp, #96]
}
    3efc:	a8c97bfd 	ldp	x29, x30, [sp], #144
    3f00:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3f04:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    3f08:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    3f0c:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    3f10:	91200084 	add	x4, x4, #0x800
    3f14:	911f0042 	add	x2, x2, #0x7c0
    3f18:	911fa000 	add	x0, x0, #0x7e8
    3f1c:	52800823 	mov	w3, #0x41                  	// #65
    3f20:	17ffff87 	b	3d3c <sptMTTKRP+0x8c>
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    3f24:	97fffecb 	bl	3a50 <sptMTTKRP_3D>
    3f28:	34fff140 	cbz	w0, 3d50 <sptMTTKRP+0xa0>
    3f2c:	12800000 	mov	w0, #0xffffffff            	// #-1
    3f30:	a90153f3 	stp	x19, x20, [sp, #16]
    3f34:	a9025bf5 	stp	x21, x22, [sp, #32]
    3f38:	a90363f7 	stp	x23, x24, [sp, #48]
    3f3c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3f40:	a90573fb 	stp	x27, x28, [sp, #80]
    3f44:	fd0033e8 	str	d8, [sp, #96]
    3f48:	97fff4a6 	bl	11e0 <exit@plt>
    3f4c:	d503201f 	nop

0000000000003f50 <sptOmpMTTKRP._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3f50:	a9b27bfd 	stp	x29, x30, [sp, #-224]!
    3f54:	910003fd 	mov	x29, sp
    3f58:	a9025bf5 	stp	x21, x22, [sp, #32]
    3f5c:	f9400c15 	ldr	x21, [x0, #24]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3f60:	b40010b5 	cbz	x21, 4174 <sptOmpMTTKRP._omp_fn.0+0x224>
    3f64:	a90153f3 	stp	x19, x20, [sp, #16]
    3f68:	aa0003f4 	mov	x20, x0
    3f6c:	97fff4ed 	bl	1320 <omp_get_num_threads@plt>
    3f70:	93407c13 	sxtw	x19, w0
    3f74:	97fff4b3 	bl	1240 <omp_get_thread_num@plt>
    3f78:	9ad30aa1 	udiv	x1, x21, x19
    3f7c:	93407c16 	sxtw	x22, w0
    3f80:	9b13d420 	msub	x0, x1, x19, x21
    3f84:	eb0002df 	cmp	x22, x0
    3f88:	54001043 	b.cc	4190 <sptOmpMTTKRP._omp_fn.0+0x240>  // b.lo, b.ul, b.last
    3f8c:	9b160036 	madd	x22, x1, x22, x0
    3f90:	8b160020 	add	x0, x1, x22
    3f94:	eb0002df 	cmp	x22, x0
    3f98:	f9004be0 	str	x0, [sp, #144]
    3f9c:	54000f22 	b.cs	4180 <sptOmpMTTKRP._omp_fn.0+0x230>  // b.hs, b.nlast
    3fa0:	910323e0 	add	x0, sp, #0xc8
    3fa4:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    3fa8:	b940429c 	ldr	w28, [x20, #64]
    3fac:	f90047e0 	str	x0, [sp, #136]
    3fb0:	f9400280 	ldr	x0, [x20]
    3fb4:	a90363f7 	stp	x23, x24, [sp, #48]
    3fb8:	51000795 	sub	w21, w28, #0x1
    3fbc:	f9400698 	ldr	x24, [x20, #8]
    3fc0:	910006b5 	add	x21, x21, #0x1
    3fc4:	f90053e0 	str	x0, [sp, #160]
		sptValueVector scratch;  // Temporary array
		sptNewValueVector(&scratch, R, R);
    3fc8:	2a1c03e0 	mov	w0, w28
    3fcc:	d37ef6b5 	lsl	x21, x21, #2
    3fd0:	a9046bf9 	stp	x25, x26, [sp, #64]
		sptConstantValueVector(&scratch, 0);

		sptIndex times_mat_index = mats_order[1];
		sptMatrix * times_mat = mats[times_mat_index];
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3fd4:	52800319 	mov	w25, #0x18                  	// #24
		sptNewValueVector(&scratch, R, R);
    3fd8:	f90057e0 	str	x0, [sp, #168]
#pragma omp parallel for schedule(static) num_threads(tk)
    3fdc:	f9400a80 	ldr	x0, [x20, #16]
    3fe0:	fd0033e8 	str	d8, [sp, #96]
    3fe4:	f9003fe0 	str	x0, [sp, #120]
    3fe8:	f9401280 	ldr	x0, [x20, #32]
    3fec:	f9004fe0 	str	x0, [sp, #152]
    3ff0:	f9401680 	ldr	x0, [x20, #40]
    3ff4:	f9005fe0 	str	x0, [sp, #184]
    3ff8:	f9401a80 	ldr	x0, [x20, #48]
    3ffc:	f9005be0 	str	x0, [sp, #176]
    4000:	29475280 	ldp	w0, w20, [x20, #56]
    4004:	b90087e0 	str	w0, [sp, #132]
		sptNewValueVector(&scratch, R, R);
    4008:	f94047f3 	ldr	x19, [sp, #136]
    400c:	f94057e1 	ldr	x1, [sp, #168]
    4010:	aa1303e0 	mov	x0, x19
    4014:	aa0103e2 	mov	x2, x1
    4018:	97fff82e 	bl	20d0 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    401c:	0f000400 	movi	v0.2s, #0x0
    4020:	aa1303e0 	mov	x0, x19
    4024:	97fff853 	bl	2170 <sptConstantValueVector>
		sptIndex times_mat_index = mats_order[1];
    4028:	f9403fe0 	ldr	x0, [sp, #120]
    402c:	d37ef6c8 	lsl	x8, x22, #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    4030:	f94053e1 	ldr	x1, [sp, #160]
		sptIndex times_mat_index = mats_order[1];
    4034:	b9400400 	ldr	w0, [x0, #4]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    4038:	f9401027 	ldr	x7, [x1, #32]
		sptIndex tmp_i = times_inds[x];
		sptValue const entry = vals[x];
    403c:	f9404fe1 	ldr	x1, [sp, #152]
		sptMatrix * times_mat = mats[times_mat_index];
    4040:	f8605b02 	ldr	x2, [x24, w0, uxtw #3]
		sptIndex tmp_i = times_inds[x];
    4044:	9bb91c00 	umaddl	x0, w0, w25, x7
		sptValue const entry = vals[x];
    4048:	bc767821 	ldr	s1, [x1, x22, lsl #2]
		sptIndex tmp_i = times_inds[x];
    404c:	f9400800 	ldr	x0, [x0, #16]
    4050:	b8767803 	ldr	w3, [x0, x22, lsl #2]
#pragma omp simd
		for(sptIndex r=0; r<R; ++r) {
    4054:	340007bc 	cbz	w28, 4148 <sptOmpMTTKRP._omp_fn.0+0x1f8>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    4058:	1b037e83 	mul	w3, w20, w3
    405c:	d2800000 	mov	x0, #0x0                   	// #0
    4060:	f9400845 	ldr	x5, [x2, #16]
    4064:	f9406fe1 	ldr	x1, [sp, #216]
    4068:	0b000062 	add	w2, w3, w0
    406c:	bc6278a0 	ldr	s0, [x5, x2, lsl #2]
    4070:	1e210800 	fmul	s0, s0, s1
    4074:	bc207820 	str	s0, [x1, x0, lsl #2]
    4078:	91000400 	add	x0, x0, #0x1
    407c:	6b00039f 	cmp	w28, w0
    4080:	54ffff48 	b.hi	4068 <sptOmpMTTKRP._omp_fn.0+0x118>  // b.pmore
		}

		for(sptIndex i=2; i<nmodes; ++i) {
    4084:	b94087e0 	ldr	w0, [sp, #132]
			times_inds = X->inds[times_mat_index].data;
			tmp_i = times_inds[x];

#pragma omp simd
			for(sptIndex r=0; r<R; ++r) {
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    4088:	d2800046 	mov	x6, #0x2                   	// #2
		for(sptIndex i=2; i<nmodes; ++i) {
    408c:	7100081f 	cmp	w0, #0x2
    4090:	54000309 	b.ls	40f0 <sptOmpMTTKRP._omp_fn.0+0x1a0>  // b.plast
    4094:	d503201f 	nop
			times_mat_index = mats_order[i];
    4098:	f9403fe0 	ldr	x0, [sp, #120]
    409c:	b8667802 	ldr	w2, [x0, x6, lsl #2]
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    40a0:	d2800000 	mov	x0, #0x0                   	// #0
			tmp_i = times_inds[x];
    40a4:	9bb91c45 	umaddl	x5, w2, w25, x7
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    40a8:	f8625b03 	ldr	x3, [x24, w2, uxtw #3]
			tmp_i = times_inds[x];
    40ac:	f94008a2 	ldr	x2, [x5, #16]
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    40b0:	f9400865 	ldr	x5, [x3, #16]
    40b4:	b8686843 	ldr	w3, [x2, x8]
    40b8:	1b037e83 	mul	w3, w20, w3
    40bc:	d503201f 	nop
    40c0:	0b000062 	add	w2, w3, w0
    40c4:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    40c8:	bc6278a1 	ldr	s1, [x5, x2, lsl #2]
    40cc:	1e210800 	fmul	s0, s0, s1
    40d0:	bc207820 	str	s0, [x1, x0, lsl #2]
    40d4:	91000400 	add	x0, x0, #0x1
    40d8:	6b00039f 	cmp	w28, w0
    40dc:	54ffff28 	b.hi	40c0 <sptOmpMTTKRP._omp_fn.0+0x170>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    40e0:	b94087e0 	ldr	w0, [sp, #132]
    40e4:	910004c6 	add	x6, x6, #0x1
    40e8:	6b06001f 	cmp	w0, w6
    40ec:	54fffd68 	b.hi	4098 <sptOmpMTTKRP._omp_fn.0+0x148>  // b.pmore
			}
		}

		sptIndex const mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    40f0:	f9405fe0 	ldr	x0, [sp, #184]
#pragma omp parallel for schedule(static) num_threads(tk)
    40f4:	d280001b 	mov	x27, #0x0                   	// #0
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    40f8:	b8767813 	ldr	w19, [x0, x22, lsl #2]
    40fc:	f9405be0 	ldr	x0, [sp, #176]
    4100:	1b137e93 	mul	w19, w20, w19
		for(sptIndex r=0; r<R; ++r) {
    4104:	8b130813 	add	x19, x0, x19, lsl #2
    4108:	14000002 	b	4110 <sptOmpMTTKRP._omp_fn.0+0x1c0>
    410c:	f9406fe1 	ldr	x1, [sp, #216]
    4110:	8b130377 	add	x23, x27, x19
#pragma omp atomic update
			mvals_row[r] += scratch.data[r];
    4114:	bc7b6828 	ldr	s8, [x1, x27]
    4118:	b94002fa 	ldr	w26, [x23]
    411c:	1e270340 	fmov	s0, w26
    4120:	aa1703e2 	mov	x2, x23
    4124:	2a1a03e0 	mov	w0, w26
    4128:	1e202900 	fadd	s0, s8, s0
    412c:	1e260001 	fmov	w1, s0
    4130:	94000370 	bl	4ef0 <__aarch64_cas4_acq_rel>
    4134:	6b00035f 	cmp	w26, w0
    4138:	54000321 	b.ne	419c <sptOmpMTTKRP._omp_fn.0+0x24c>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    413c:	9100137b 	add	x27, x27, #0x4
    4140:	eb15037f 	cmp	x27, x21
    4144:	54fffe41 	b.ne	410c <sptOmpMTTKRP._omp_fn.0+0x1bc>  // b.any
		}

		sptFreeValueVector(&scratch);
    4148:	f94047e0 	ldr	x0, [sp, #136]
    414c:	910006d6 	add	x22, x22, #0x1
    4150:	97fff8b4 	bl	2420 <sptFreeValueVector>
    4154:	f9404be0 	ldr	x0, [sp, #144]
    4158:	eb16001f 	cmp	x0, x22
    415c:	54fff568 	b.hi	4008 <sptOmpMTTKRP._omp_fn.0+0xb8>  // b.pmore
    4160:	fd4033e8 	ldr	d8, [sp, #96]
    4164:	a94153f3 	ldp	x19, x20, [sp, #16]
    4168:	a94363f7 	ldp	x23, x24, [sp, #48]
    416c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4170:	a94573fb 	ldp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    4174:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4178:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    417c:	d65f03c0 	ret
    4180:	a94153f3 	ldp	x19, x20, [sp, #16]
    4184:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4188:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    418c:	d65f03c0 	ret
    4190:	91000421 	add	x1, x1, #0x1
    4194:	d2800000 	mov	x0, #0x0                   	// #0
    4198:	17ffff7d 	b	3f8c <sptOmpMTTKRP._omp_fn.0+0x3c>
			mvals_row[r] += scratch.data[r];
    419c:	2a0003fa 	mov	w26, w0
    41a0:	17ffffdf 	b	411c <sptOmpMTTKRP._omp_fn.0+0x1cc>
    41a4:	d503201f 	nop
    41a8:	d503201f 	nop
    41ac:	d503201f 	nop

00000000000041b0 <sptOmpMTTKRP_3D._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    41b0:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    41b4:	910003fd 	mov	x29, sp
    41b8:	a9025bf5 	stp	x21, x22, [sp, #32]
    41bc:	f9400016 	ldr	x22, [x0]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    41c0:	b40009f6 	cbz	x22, 42fc <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
    41c4:	a90153f3 	stp	x19, x20, [sp, #16]
    41c8:	aa0003f3 	mov	x19, x0
    41cc:	a90363f7 	stp	x23, x24, [sp, #48]
    41d0:	97fff454 	bl	1320 <omp_get_num_threads@plt>
    41d4:	93407c14 	sxtw	x20, w0
    41d8:	97fff41a 	bl	1240 <omp_get_thread_num@plt>
    41dc:	9ad40ad7 	udiv	x23, x22, x20
    41e0:	93407c15 	sxtw	x21, w0
    41e4:	9b14dae0 	msub	x0, x23, x20, x22
    41e8:	eb0002bf 	cmp	x21, x0
    41ec:	54000983 	b.cc	431c <sptOmpMTTKRP_3D._omp_fn.0+0x16c>  // b.lo, b.ul, b.last
    41f0:	9b1502f5 	madd	x21, x23, x21, x0
    41f4:	8b1502e0 	add	x0, x23, x21
    41f8:	eb0002bf 	cmp	x21, x0
    41fc:	f9003be0 	str	x0, [sp, #112]
    4200:	54000842 	b.cs	4308 <sptOmpMTTKRP_3D._omp_fn.0+0x158>  // b.hs, b.nlast
    4204:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    4208:	a940ee61 	ldp	x1, x27, [x19, #8]
    420c:	29480276 	ldp	w22, w0, [x19, #64]
    4210:	f9004fe1 	str	x1, [sp, #152]
    4214:	f9400e61 	ldr	x1, [x19, #24]
    4218:	f9401278 	ldr	x24, [x19, #32]
    421c:	b90097e0 	str	w0, [sp, #148]
    4220:	f9401a7c 	ldr	x28, [x19, #48]
    4224:	f90047e1 	str	x1, [sp, #136]
    4228:	f9401661 	ldr	x1, [x19, #40]
    422c:	f90043e1 	str	x1, [sp, #128]
    4230:	f9401e61 	ldr	x1, [x19, #56]
    4234:	f9003fe1 	str	x1, [sp, #120]
    4238:	34000780 	cbz	w0, 4328 <sptOmpMTTKRP_3D._omp_fn.0+0x178>
    423c:	a9046bf9 	stp	x25, x26, [sp, #64]
    4240:	6d0627e8 	stp	d8, d9, [sp, #96]
    4244:	d503201f 	nop
		sptIndex tmp_i_2 = times_inds_2[x];
		sptValue entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4248:	f9403fe0 	ldr	x0, [sp, #120]
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    424c:	b8757b62 	ldr	w2, [x27, x21, lsl #2]
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4250:	b8757813 	ldr	w19, [x0, x21, lsl #2]
    4254:	f94043e0 	ldr	x0, [sp, #128]
    4258:	1b027ec2 	mul	w2, w22, w2
    425c:	1b137ed3 	mul	w19, w22, w19
    4260:	b8757814 	ldr	w20, [x0, x21, lsl #2]
    4264:	2a1303f9 	mov	w25, w19
		sptValue entry = vals[x];
    4268:	f9404fe0 	ldr	x0, [sp, #152]
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    426c:	1b147ed4 	mul	w20, w22, w20
		sptValue entry = vals[x];
    4270:	bc757809 	ldr	s9, [x0, x21, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    4274:	4b130294 	sub	w20, w20, w19
    4278:	b94097e0 	ldr	w0, [sp, #148]
    427c:	0b130013 	add	w19, w0, w19
    4280:	f94047e0 	ldr	x0, [sp, #136]
    4284:	8b020817 	add	x23, x0, x2, lsl #2
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4288:	f9400b04 	ldr	x4, [x24, #16]
    428c:	0b190280 	add	w0, w20, w25
    4290:	f9400b81 	ldr	x1, [x28, #16]
    4294:	bc607888 	ldr	s8, [x4, x0, lsl #2]
    4298:	bc795820 	ldr	s0, [x1, w25, uxtw #2]
    429c:	1e200908 	fmul	s8, s8, s0
    42a0:	1e290908 	fmul	s8, s8, s9
    42a4:	b94002fa 	ldr	w26, [x23]
    42a8:	1e270340 	fmov	s0, w26
    42ac:	aa1703e2 	mov	x2, x23
    42b0:	2a1a03e0 	mov	w0, w26
    42b4:	1e202900 	fadd	s0, s8, s0
    42b8:	1e260001 	fmov	w1, s0
    42bc:	9400030d 	bl	4ef0 <__aarch64_cas4_acq_rel>
    42c0:	6b00035f 	cmp	w26, w0
    42c4:	540003a1 	b.ne	4338 <sptOmpMTTKRP_3D._omp_fn.0+0x188>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    42c8:	11000739 	add	w25, w25, #0x1
    42cc:	910012f7 	add	x23, x23, #0x4
    42d0:	6b19027f 	cmp	w19, w25
    42d4:	54fffda1 	b.ne	4288 <sptOmpMTTKRP_3D._omp_fn.0+0xd8>  // b.any
    42d8:	f9403be0 	ldr	x0, [sp, #112]
    42dc:	910006b5 	add	x21, x21, #0x1
    42e0:	eb15001f 	cmp	x0, x21
    42e4:	54fffb21 	b.ne	4248 <sptOmpMTTKRP_3D._omp_fn.0+0x98>  // b.any
    42e8:	6d4627e8 	ldp	d8, d9, [sp, #96]
    42ec:	a94153f3 	ldp	x19, x20, [sp, #16]
    42f0:	a94363f7 	ldp	x23, x24, [sp, #48]
    42f4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    42f8:	a94573fb 	ldp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    42fc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4300:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4304:	d65f03c0 	ret
    4308:	a94153f3 	ldp	x19, x20, [sp, #16]
    430c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4310:	a94363f7 	ldp	x23, x24, [sp, #48]
    4314:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4318:	d65f03c0 	ret
    431c:	910006f7 	add	x23, x23, #0x1
    4320:	d2800000 	mov	x0, #0x0                   	// #0
    4324:	17ffffb3 	b	41f0 <sptOmpMTTKRP_3D._omp_fn.0+0x40>
    4328:	a94153f3 	ldp	x19, x20, [sp, #16]
    432c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4330:	a94573fb 	ldp	x27, x28, [sp, #80]
    4334:	17fffff2 	b	42fc <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4338:	2a0003fa 	mov	w26, w0
    433c:	17ffffdb 	b	42a8 <sptOmpMTTKRP_3D._omp_fn.0+0xf8>

0000000000004340 <sptOmpMTTKRP_3D>:
{
    4340:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    4344:	910003fd 	mov	x29, sp
    4348:	a90153f3 	stp	x19, x20, [sp, #16]
    434c:	aa0103f3 	mov	x19, x1
    4350:	aa0003f4 	mov	x20, x0
	sptAssert(nmodes ==3);
    4354:	b9400001 	ldr	w1, [x0]
{
    4358:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex const stride = mats[0]->stride;
    435c:	f9400260 	ldr	x0, [x19]
	sptAssert(nmodes ==3);
    4360:	71000c3f 	cmp	w1, #0x3
{
    4364:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptNnzIndex const nnz = X->nnz;
    4368:	a9416a85 	ldp	x5, x26, [x20, #16]
	sptIndex const stride = mats[0]->stride;
    436c:	b9400c18 	ldr	w24, [x0, #12]
	sptValue const * const restrict vals = X->values.data;
    4370:	f9401e99 	ldr	x25, [x20, #56]
	sptAssert(nmodes ==3);
    4374:	a9025bf5 	stp	x21, x22, [sp, #32]
    4378:	54000e41 	b.ne	4540 <sptOmpMTTKRP_3D+0x200>  // b.any
    437c:	aa0203f6 	mov	x22, x2
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4380:	f9400e66 	ldr	x6, [x19, #24]
    4384:	b9400402 	ldr	w2, [x0, #4]
    4388:	b94004c1 	ldr	w1, [x6, #4]
    438c:	6b02003f 	cmp	w1, w2
    4390:	54000ac1 	b.ne	44e8 <sptOmpMTTKRP_3D+0x1a8>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    4394:	b9400002 	ldr	w2, [x0]
    4398:	b94000a0 	ldr	w0, [x5]
    439c:	6b00005f 	cmp	w2, w0
    43a0:	54000ba1 	b.ne	4514 <sptOmpMTTKRP_3D+0x1d4>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    43a4:	f9400660 	ldr	x0, [x19, #8]
    43a8:	b9400402 	ldr	w2, [x0, #4]
    43ac:	6b02003f 	cmp	w1, w2
    43b0:	540009c1 	b.ne	44e8 <sptOmpMTTKRP_3D+0x1a8>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    43b4:	b9400002 	ldr	w2, [x0]
    43b8:	b94004a0 	ldr	w0, [x5, #4]
    43bc:	6b00005f 	cmp	w2, w0
    43c0:	54000aa1 	b.ne	4514 <sptOmpMTTKRP_3D+0x1d4>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    43c4:	f9400a60 	ldr	x0, [x19, #16]
    43c8:	b9400402 	ldr	w2, [x0, #4]
    43cc:	6b01005f 	cmp	w2, w1
    43d0:	540008c1 	b.ne	44e8 <sptOmpMTTKRP_3D+0x1a8>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    43d4:	b9400001 	ldr	w1, [x0]
    43d8:	b94008a0 	ldr	w0, [x5, #8]
    43dc:	6b00003f 	cmp	w1, w0
    43e0:	540009a1 	b.ne	4514 <sptOmpMTTKRP_3D+0x1d4>  // b.any
	sptIndex const tmpI = mats[mode]->nrows;
    43e4:	f8635a60 	ldr	x0, [x19, w3, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    43e8:	52800315 	mov	w21, #0x18                  	// #24
    43ec:	2a0403f7 	mov	w23, w4
    43f0:	a90573fb 	stp	x27, x28, [sp, #80]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    43f4:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    43f8:	f9401282 	ldr	x2, [x20, #32]
	sptIndex const R = mats[mode]->ncols;
    43fc:	29407005 	ldp	w5, w28, [x0]
	sptValue * const restrict mvals = mats[nmodes]->values;
    4400:	f94008db 	ldr	x27, [x6, #16]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    4404:	9bb50863 	umaddl	x3, w3, w21, x2
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4408:	1b057f02 	mul	w2, w24, w5
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    440c:	f9400864 	ldr	x4, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4410:	aa1b03e0 	mov	x0, x27
    4414:	fd0033e8 	str	d8, [sp, #96]
    4418:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    441c:	f9003fe4 	str	x4, [sp, #120]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4420:	97fff3ac 	bl	12d0 <memset@plt>
	sptIndex times_mat_index_2 = mats_order[2];
    4424:	29408ec5 	ldp	w5, w3, [x22, #4]
	sptNewTimer(&timer, 0);
    4428:	52800001 	mov	w1, #0x0                   	// #0
    442c:	910203e0 	add	x0, sp, #0x80
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4430:	f9401286 	ldr	x6, [x20, #32]
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    4434:	f8655a76 	ldr	x22, [x19, w5, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4438:	9bb518a5 	umaddl	x5, w5, w21, x6
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    443c:	f8635a74 	ldr	x20, [x19, w3, uxtw #3]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    4440:	9bb51863 	umaddl	x3, w3, w21, x6
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4444:	f94008b5 	ldr	x21, [x5, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    4448:	f9400873 	ldr	x19, [x3, #16]
	sptNewTimer(&timer, 0);
    444c:	940000bd 	bl	4740 <sptNewTimer>
	sptStartTimer(timer);
    4450:	f94043e0 	ldr	x0, [sp, #128]
    4454:	940000db 	bl	47c0 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    4458:	f9403fe4 	ldr	x4, [sp, #120]
    445c:	90000000 	adrp	x0, 4000 <sptOmpMTTKRP._omp_fn.0+0xb0>
    4460:	2a1703e2 	mov	w2, w23
    4464:	52800003 	mov	w3, #0x0                   	// #0
    4468:	910223e1 	add	x1, sp, #0x88
    446c:	a908e7fa 	stp	x26, x25, [sp, #136]
    4470:	9106c000 	add	x0, x0, #0x1b0
    4474:	a90ad7f6 	stp	x22, x21, [sp, #168]
    4478:	a909efe4 	stp	x4, x27, [sp, #152]
    447c:	a90bcff4 	stp	x20, x19, [sp, #184]
    4480:	291973f8 	stp	w24, w28, [sp, #200]
    4484:	97fff3ef 	bl	1440 <GOMP_parallel@plt>
		}
	}
	sptStopTimer(timer);
    4488:	f94043e0 	ldr	x0, [sp, #128]
    448c:	940000fd 	bl	4880 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    4490:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4494:	91218021 	add	x1, x1, #0x860
    4498:	f94043e0 	ldr	x0, [sp, #128]
    449c:	94000139 	bl	4980 <sptPrintElapsedTime>
    44a0:	1e604008 	fmov	d8, d0

	sptFreeTimer(timer);
    44a4:	f94043e0 	ldr	x0, [sp, #128]
    44a8:	94000176 	bl	4a80 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    44ac:	1e604100 	fmov	d0, d8
    44b0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    44b4:	91208000 	add	x0, x0, #0x820
    44b8:	97fff3da 	bl	1420 <printf@plt>
	printf("\n");
    44bc:	52800140 	mov	w0, #0xa                   	// #10
    44c0:	97fff3e4 	bl	1450 <putchar@plt>

	return 0;
    44c4:	fd4033e8 	ldr	d8, [sp, #96]
    44c8:	52800000 	mov	w0, #0x0                   	// #0
    44cc:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    44d0:	a94153f3 	ldp	x19, x20, [sp, #16]
    44d4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    44d8:	a94363f7 	ldp	x23, x24, [sp, #48]
    44dc:	a9446bf9 	ldp	x25, x26, [sp, #64]
    44e0:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    44e4:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    44e8:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    44ec:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    44f0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    44f4:	911e6084 	add	x4, x4, #0x798
    44f8:	91218000 	add	x0, x0, #0x860
    44fc:	9120e042 	add	x2, x2, #0x838
    4500:	528012c3 	mov	w3, #0x96                  	// #150
    4504:	52800041 	mov	w1, #0x2                   	// #2
    4508:	940001de 	bl	4c80 <spt_ComplainError>
    450c:	52800040 	mov	w0, #0x2                   	// #2
    4510:	17fffff0 	b	44d0 <sptOmpMTTKRP_3D+0x190>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4514:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    4518:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    451c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    4520:	91200084 	add	x4, x4, #0x800
    4524:	91218000 	add	x0, x0, #0x860
    4528:	9120e042 	add	x2, x2, #0x838
    452c:	52801323 	mov	w3, #0x99                  	// #153
    4530:	52800041 	mov	w1, #0x2                   	// #2
    4534:	940001d3 	bl	4c80 <spt_ComplainError>
    4538:	52800040 	mov	w0, #0x2                   	// #2
    453c:	17ffffe5 	b	44d0 <sptOmpMTTKRP_3D+0x190>
	sptAssert(nmodes ==3);
    4540:	12800000 	mov	w0, #0xffffffff            	// #-1
    4544:	a90573fb 	stp	x27, x28, [sp, #80]
    4548:	fd0033e8 	str	d8, [sp, #96]
    454c:	97fff325 	bl	11e0 <exit@plt>

0000000000004550 <sptOmpMTTKRP>:
{
    4550:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    4554:	910003fd 	mov	x29, sp
    4558:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const nmodes = X->nmodes;
    455c:	b9400016 	ldr	w22, [x0]
	if(nmodes == 3) {
    4560:	71000edf 	cmp	w22, #0x3
    4564:	54000dc0 	b.eq	471c <sptOmpMTTKRP+0x1cc>  // b.none
	sptNnzIndex const nnz = X->nnz;
    4568:	a90153f3 	stp	x19, x20, [sp, #16]
    456c:	aa0203f5 	mov	x21, x2
    4570:	aa0103f3 	mov	x19, x1
    4574:	2a0303e9 	mov	w9, w3
    4578:	2a0403f4 	mov	w20, w4
    457c:	a90363f7 	stp	x23, x24, [sp, #48]
    4580:	aa0003f7 	mov	x23, x0
	sptValue * const restrict mvals = mats[nmodes]->values;
    4584:	f8765820 	ldr	x0, [x1, w22, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    4588:	f9400ae8 	ldr	x8, [x23, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    458c:	340003d6 	cbz	w22, 4604 <sptOmpMTTKRP+0xb4>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4590:	d2800005 	mov	x5, #0x0                   	// #0
    4594:	b9400401 	ldr	w1, [x0, #4]
    4598:	14000008 	b	45b8 <sptOmpMTTKRP+0x68>
		if(mats[i]->nrows != ndims[i]) {
    459c:	b94000c7 	ldr	w7, [x6]
    45a0:	b8657906 	ldr	w6, [x8, x5, lsl #2]
    45a4:	6b0600ff 	cmp	w7, w6
    45a8:	54000aa1 	b.ne	46fc <sptOmpMTTKRP+0x1ac>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    45ac:	910004a5 	add	x5, x5, #0x1
    45b0:	6b0502df 	cmp	w22, w5
    45b4:	54000289 	b.ls	4604 <sptOmpMTTKRP+0xb4>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    45b8:	f8657a66 	ldr	x6, [x19, x5, lsl #3]
    45bc:	b94004c7 	ldr	w7, [x6, #4]
    45c0:	6b0100ff 	cmp	w7, w1
    45c4:	54fffec0 	b.eq	459c <sptOmpMTTKRP+0x4c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    45c8:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    45cc:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    45d0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    45d4:	911e6084 	add	x4, x4, #0x798
    45d8:	9120e042 	add	x2, x2, #0x838
    45dc:	91218000 	add	x0, x0, #0x860
    45e0:	52800823 	mov	w3, #0x41                  	// #65
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    45e4:	52800041 	mov	w1, #0x2                   	// #2
    45e8:	940001a6 	bl	4c80 <spt_ComplainError>
    45ec:	a94153f3 	ldp	x19, x20, [sp, #16]
    45f0:	52800040 	mov	w0, #0x2                   	// #2
    45f4:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    45f8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    45fc:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4600:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    4604:	f8695a64 	ldr	x4, [x19, w9, uxtw #3]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4608:	52800305 	mov	w5, #0x18                  	// #24
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    460c:	52800001 	mov	w1, #0x0                   	// #0
    4610:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const stride = mats[0]->stride;
    4614:	f9400262 	ldr	x2, [x19]
    4618:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex const R = mats[mode]->ncols;
    461c:	b940049a 	ldr	w26, [x4, #4]
	sptIndex const stride = mats[0]->stride;
    4620:	b9400c58 	ldr	w24, [x2, #12]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4624:	b9400082 	ldr	w2, [x4]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4628:	a9418ee4 	ldp	x4, x3, [x23, #24]
	sptValue * const restrict mvals = mats[nmodes]->values;
    462c:	f9400819 	ldr	x25, [x0, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4630:	1b027f02 	mul	w2, w24, w2
	sptIndex const * const mode_ind = X->inds[mode].data;
    4634:	9ba50d23 	umaddl	x3, w9, w5, x3
	sptNnzIndex const nnz = X->nnz;
    4638:	f9003fe4 	str	x4, [sp, #120]
	sptValue const * const vals = X->values.data;
    463c:	f9401efc 	ldr	x28, [x23, #56]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4640:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const mode_ind = X->inds[mode].data;
    4644:	f940087b 	ldr	x27, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4648:	aa1903e0 	mov	x0, x25
    464c:	fd0033e8 	str	d8, [sp, #96]
    4650:	97fff320 	bl	12d0 <memset@plt>
	sptNewTimer(&timer, 0);
    4654:	52800001 	mov	w1, #0x0                   	// #0
    4658:	910203e0 	add	x0, sp, #0x80
    465c:	94000039 	bl	4740 <sptNewTimer>
	sptStartTimer(timer);
    4660:	f94043e0 	ldr	x0, [sp, #128]
    4664:	94000057 	bl	47c0 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    4668:	f9403fe4 	ldr	x4, [sp, #120]
    466c:	f0ffffe0 	adrp	x0, 3000 <sptLoadSparseTensor+0x7a0>
    4670:	2a1403e2 	mov	w2, w20
    4674:	52800003 	mov	w3, #0x0                   	// #0
    4678:	910223e1 	add	x1, sp, #0x88
    467c:	a908cff7 	stp	x23, x19, [sp, #136]
    4680:	913d4000 	add	x0, x0, #0xf50
    4684:	a90aeffc 	stp	x28, x27, [sp, #168]
    4688:	a90993f5 	stp	x21, x4, [sp, #152]
    468c:	f9005ff9 	str	x25, [sp, #184]
    4690:	291863f6 	stp	w22, w24, [sp, #192]
    4694:	b900cbfa 	str	w26, [sp, #200]
    4698:	97fff36a 	bl	1440 <GOMP_parallel@plt>
	sptStopTimer(timer);
    469c:	f94043e0 	ldr	x0, [sp, #128]
    46a0:	94000078 	bl	4880 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    46a4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    46a8:	91218021 	add	x1, x1, #0x860
    46ac:	f94043e0 	ldr	x0, [sp, #128]
    46b0:	940000b4 	bl	4980 <sptPrintElapsedTime>
    46b4:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    46b8:	f94043e0 	ldr	x0, [sp, #128]
    46bc:	940000f1 	bl	4a80 <sptFreeTimer>
	printf("[Total time]: %lf\n", total_time);
    46c0:	1e604100 	fmov	d0, d8
    46c4:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    46c8:	91208000 	add	x0, x0, #0x820
    46cc:	97fff355 	bl	1420 <printf@plt>
	printf("\n");
    46d0:	52800140 	mov	w0, #0xa                   	// #10
    46d4:	97fff35f 	bl	1450 <putchar@plt>
	return 0;
    46d8:	52800000 	mov	w0, #0x0                   	// #0
    46dc:	a94153f3 	ldp	x19, x20, [sp, #16]
}
    46e0:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    46e4:	a94363f7 	ldp	x23, x24, [sp, #48]
    46e8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    46ec:	a94573fb 	ldp	x27, x28, [sp, #80]
    46f0:	fd4033e8 	ldr	d8, [sp, #96]
}
    46f4:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    46f8:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    46fc:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    4700:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4704:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    4708:	91200084 	add	x4, x4, #0x800
    470c:	9120e042 	add	x2, x2, #0x838
    4710:	91218000 	add	x0, x0, #0x860
    4714:	52800883 	mov	w3, #0x44                  	// #68
    4718:	17ffffb3 	b	45e4 <sptOmpMTTKRP+0x94>
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    471c:	97ffff09 	bl	4340 <sptOmpMTTKRP_3D>
    4720:	34fff6c0 	cbz	w0, 45f8 <sptOmpMTTKRP+0xa8>
    4724:	12800000 	mov	w0, #0xffffffff            	// #-1
    4728:	a90153f3 	stp	x19, x20, [sp, #16]
    472c:	a90363f7 	stp	x23, x24, [sp, #48]
    4730:	a9046bf9 	stp	x25, x26, [sp, #64]
    4734:	a90573fb 	stp	x27, x28, [sp, #80]
    4738:	fd0033e8 	str	d8, [sp, #96]
    473c:	97fff2a9 	bl	11e0 <exit@plt>

0000000000004740 <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    4740:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4744:	910003fd 	mov	x29, sp
    4748:	a90153f3 	stp	x19, x20, [sp, #16]
    474c:	aa0003f4 	mov	x20, x0
    4750:	2a0103f3 	mov	w19, w1
	*timer = (sptTimer) malloc(sizeof **timer);
    4754:	d2800500 	mov	x0, #0x28                  	// #40
    4758:	97fff2ca 	bl	1280 <malloc@plt>
	(*timer)->use_cuda = use_cuda;
    475c:	b9000013 	str	w19, [x0]
	*timer = (sptTimer) malloc(sizeof **timer);
    4760:	f9000280 	str	x0, [x20]
	if(use_cuda) {
    4764:	350000b3 	cbnz	w19, 4778 <sptNewTimer+0x38>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	return 0;
}
    4768:	2a1303e0 	mov	w0, w19
    476c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4770:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4774:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4778:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    477c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4780:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    4784:	9121e084 	add	x4, x4, #0x878
    4788:	91228042 	add	x2, x2, #0x8a0
    478c:	91232000 	add	x0, x0, #0x8c8
    4790:	528004e3 	mov	w3, #0x27                  	// #39
    4794:	52800061 	mov	w1, #0x3                   	// #3
    4798:	72a00041 	movk	w1, #0x2, lsl #16
    479c:	52800073 	mov	w19, #0x3                   	// #3
    47a0:	72a00053 	movk	w19, #0x2, lsl #16
    47a4:	94000137 	bl	4c80 <spt_ComplainError>
}
    47a8:	2a1303e0 	mov	w0, w19
    47ac:	a94153f3 	ldp	x19, x20, [sp, #16]
    47b0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    47b4:	d65f03c0 	ret
    47b8:	d503201f 	nop
    47bc:	d503201f 	nop

00000000000047c0 <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    47c0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    47c4:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    47c8:	b9400001 	ldr	w1, [x0]
    47cc:	350000e1 	cbnz	w1, 47e8 <sptStartTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    47d0:	91002001 	add	x1, x0, #0x8
    47d4:	52800020 	mov	w0, #0x1                   	// #1
    47d8:	97fff28e 	bl	1210 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    47dc:	35000220 	cbnz	w0, 4820 <sptStartTimer+0x60>
	}
	return 0;
}
    47e0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    47e4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    47e8:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    47ec:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    47f0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    47f4:	9121e084 	add	x4, x4, #0x878
    47f8:	91228042 	add	x2, x2, #0x8a0
    47fc:	91232000 	add	x0, x0, #0x8c8
    4800:	528005e3 	mov	w3, #0x2f                  	// #47
    4804:	52800061 	mov	w1, #0x3                   	// #3
    4808:	72a00041 	movk	w1, #0x2, lsl #16
    480c:	9400011d 	bl	4c80 <spt_ComplainError>
}
    4810:	a8c27bfd 	ldp	x29, x30, [sp], #32
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4814:	52800060 	mov	w0, #0x3                   	// #3
    4818:	72a00040 	movk	w0, #0x2, lsl #16
}
    481c:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    4820:	a90153f3 	stp	x19, x20, [sp, #16]
    4824:	97fff303 	bl	1430 <__errno_location@plt>
    4828:	aa0003f3 	mov	x19, x0
	return 0;
    482c:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    4830:	b9400274 	ldr	w20, [x19]
    4834:	3140429f 	cmn	w20, #0x10, lsl #12
    4838:	54000061 	b.ne	4844 <sptStartTimer+0x84>  // b.any
    483c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4840:	17ffffe8 	b	47e0 <sptStartTimer+0x20>
    4844:	2a1403e0 	mov	w0, w20
    4848:	97fff2ba 	bl	1330 <strerror@plt>
    484c:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    4850:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4854:	aa0003e4 	mov	x4, x0
    4858:	11404281 	add	w1, w20, #0x10, lsl #12
    485c:	91232060 	add	x0, x3, #0x8c8
    4860:	91228042 	add	x2, x2, #0x8a0
    4864:	52800643 	mov	w3, #0x32                  	// #50
    4868:	94000106 	bl	4c80 <spt_ComplainError>
    486c:	b9400260 	ldr	w0, [x19]
    4870:	a94153f3 	ldp	x19, x20, [sp, #16]
    4874:	11404000 	add	w0, w0, #0x10, lsl #12
    4878:	17ffffda 	b	47e0 <sptStartTimer+0x20>
    487c:	d503201f 	nop

0000000000004880 <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    4880:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4884:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    4888:	b9400001 	ldr	w1, [x0]
    488c:	350000e1 	cbnz	w1, 48a8 <sptStopTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    4890:	91006001 	add	x1, x0, #0x18
    4894:	52800020 	mov	w0, #0x1                   	// #1
    4898:	97fff25e 	bl	1210 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    489c:	35000220 	cbnz	w0, 48e0 <sptStopTimer+0x60>
	}
	return 0;
}
    48a0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    48a4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    48a8:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    48ac:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    48b0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    48b4:	9121e084 	add	x4, x4, #0x878
    48b8:	91228042 	add	x2, x2, #0x8a0
    48bc:	91232000 	add	x0, x0, #0x8c8
    48c0:	52800743 	mov	w3, #0x3a                  	// #58
    48c4:	52800061 	mov	w1, #0x3                   	// #3
    48c8:	72a00041 	movk	w1, #0x2, lsl #16
    48cc:	940000ed 	bl	4c80 <spt_ComplainError>
}
    48d0:	a8c27bfd 	ldp	x29, x30, [sp], #32
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    48d4:	52800060 	mov	w0, #0x3                   	// #3
    48d8:	72a00040 	movk	w0, #0x2, lsl #16
}
    48dc:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    48e0:	a90153f3 	stp	x19, x20, [sp, #16]
    48e4:	97fff2d3 	bl	1430 <__errno_location@plt>
    48e8:	aa0003f3 	mov	x19, x0
	return 0;
    48ec:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    48f0:	b9400274 	ldr	w20, [x19]
    48f4:	3140429f 	cmn	w20, #0x10, lsl #12
    48f8:	54000061 	b.ne	4904 <sptStopTimer+0x84>  // b.any
    48fc:	a94153f3 	ldp	x19, x20, [sp, #16]
    4900:	17ffffe8 	b	48a0 <sptStopTimer+0x20>
    4904:	2a1403e0 	mov	w0, w20
    4908:	97fff28a 	bl	1330 <strerror@plt>
    490c:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    4910:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4914:	aa0003e4 	mov	x4, x0
    4918:	11404281 	add	w1, w20, #0x10, lsl #12
    491c:	91232060 	add	x0, x3, #0x8c8
    4920:	91228042 	add	x2, x2, #0x8a0
    4924:	528007a3 	mov	w3, #0x3d                  	// #61
    4928:	940000d6 	bl	4c80 <spt_ComplainError>
    492c:	b9400260 	ldr	w0, [x19]
    4930:	a94153f3 	ldp	x19, x20, [sp, #16]
    4934:	11404000 	add	w0, w0, #0x10, lsl #12
    4938:	17ffffda 	b	48a0 <sptStopTimer+0x20>
    493c:	d503201f 	nop

0000000000004940 <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
	if(timer->use_cuda) {
    4940:	b9400001 	ldr	w1, [x0]
    4944:	35000161 	cbnz	w1, 4970 <sptElapsedTime+0x30>
		return NAN;
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4948:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    494c:	a9408c04 	ldp	x4, x3, [x0, #8]
    4950:	fd420c22 	ldr	d2, [x1, #1048]
    4954:	a9418801 	ldp	x1, x2, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4958:	cb040020 	sub	x0, x1, x4
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    495c:	cb030041 	sub	x1, x2, x3
    4960:	9e620000 	scvtf	d0, x0
    4964:	9e620021 	scvtf	d1, x1
    4968:	1f420020 	fmadd	d0, d1, d2, d0
	}
}
    496c:	d65f03c0 	ret
		return NAN;
    4970:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4974:	9e670000 	fmov	d0, x0
}
    4978:	d65f03c0 	ret
    497c:	d503201f 	nop

0000000000004980 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4980:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4984:	910003fd 	mov	x29, sp
    4988:	aa0103e2 	mov	x2, x1
	if(timer->use_cuda) {
    498c:	b9400001 	ldr	w1, [x0]
double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4990:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4994:	350002a1 	cbnz	w1, 49e8 <sptPrintElapsedTime+0x68>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4998:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    499c:	a9409005 	ldp	x5, x4, [x0, #8]
    49a0:	fd420c21 	ldr	d1, [x1, #1048]
    49a4:	a9418403 	ldp	x3, x1, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    49a8:	cb050060 	sub	x0, x3, x5
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    49ac:	cb040021 	sub	x1, x1, x4
    49b0:	9e620008 	scvtf	d8, x0
    49b4:	9e620020 	scvtf	d0, x1
    49b8:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer);
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    49bc:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    49c0:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    49c4:	1e604100 	fmov	d0, d8
    49c8:	f947e000 	ldr	x0, [x0, #4032]
    49cc:	91236021 	add	x1, x1, #0x8d8
    49d0:	f9400000 	ldr	x0, [x0]
    49d4:	97fff2a3 	bl	1460 <fprintf@plt>
	return elapsed_time;
}
    49d8:	1e604100 	fmov	d0, d8
    49dc:	fd400be8 	ldr	d8, [sp, #16]
    49e0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    49e4:	d65f03c0 	ret
		return NAN;
    49e8:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    49ec:	9e670008 	fmov	d8, x0
    49f0:	17fffff3 	b	49bc <sptPrintElapsedTime+0x3c>
    49f4:	d503201f 	nop
    49f8:	d503201f 	nop
    49fc:	d503201f 	nop

0000000000004a00 <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4a00:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4a04:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    4a08:	b9400003 	ldr	w3, [x0]
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4a0c:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4a10:	350002e3 	cbnz	w3, 4a6c <sptPrintAverageElapsedTime+0x6c>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4a14:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    4a18:	a9409406 	ldp	x6, x5, [x0, #8]
    4a1c:	fd420c61 	ldr	d1, [x3, #1048]
    4a20:	a9419003 	ldp	x3, x4, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4a24:	cb060060 	sub	x0, x3, x6
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4a28:	cb050083 	sub	x3, x4, x5
    4a2c:	9e620008 	scvtf	d8, x0
    4a30:	9e620060 	scvtf	d0, x3
    4a34:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer) / niters;
    4a38:	1e620020 	scvtf	d0, w1
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4a3c:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    4a40:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4a44:	f947e000 	ldr	x0, [x0, #4032]
    4a48:	91236021 	add	x1, x1, #0x8d8
	double elapsed_time = sptElapsedTime(timer) / niters;
    4a4c:	1e601908 	fdiv	d8, d8, d0
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4a50:	f9400000 	ldr	x0, [x0]
    4a54:	1e604100 	fmov	d0, d8
    4a58:	97fff282 	bl	1460 <fprintf@plt>
	return elapsed_time;
}
    4a5c:	1e604100 	fmov	d0, d8
    4a60:	fd400be8 	ldr	d8, [sp, #16]
    4a64:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4a68:	d65f03c0 	ret
		return NAN;
    4a6c:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4a70:	9e670008 	fmov	d8, x0
    4a74:	17fffff1 	b	4a38 <sptPrintAverageElapsedTime+0x38>
    4a78:	d503201f 	nop
    4a7c:	d503201f 	nop

0000000000004a80 <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    4a80:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4a84:	910003fd 	mov	x29, sp
    4a88:	f9000bf3 	str	x19, [sp, #16]
	if(timer->use_cuda) {
    4a8c:	b9400013 	ldr	w19, [x0]
    4a90:	350000d3 	cbnz	w19, 4aa8 <sptFreeTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	free(timer);
    4a94:	97fff24b 	bl	13c0 <free@plt>
	return 0;
}
    4a98:	2a1303e0 	mov	w0, w19
    4a9c:	f9400bf3 	ldr	x19, [sp, #16]
    4aa0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4aa4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4aa8:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x30>
    4aac:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4ab0:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x30>
    4ab4:	9121e084 	add	x4, x4, #0x878
    4ab8:	91228042 	add	x2, x2, #0x8a0
    4abc:	91232000 	add	x0, x0, #0x8c8
    4ac0:	52800b63 	mov	w3, #0x5b                  	// #91
    4ac4:	52800061 	mov	w1, #0x3                   	// #3
    4ac8:	72a00041 	movk	w1, #0x2, lsl #16
    4acc:	52800073 	mov	w19, #0x3                   	// #3
    4ad0:	72a00053 	movk	w19, #0x2, lsl #16
    4ad4:	9400006b 	bl	4c80 <spt_ComplainError>
}
    4ad8:	2a1303e0 	mov	w0, w19
    4adc:	f9400bf3 	ldr	x19, [sp, #16]
    4ae0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4ae4:	d65f03c0 	ret
    4ae8:	d503201f 	nop
    4aec:	d503201f 	nop

0000000000004af0 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4af0:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
    4af4:	910003fd 	mov	x29, sp
    4af8:	a9025bf5 	stp	x21, x22, [sp, #32]
    4afc:	aa0003f6 	mov	x22, x0
    4b00:	aa0103f5 	mov	x21, x1
	int iores;
	sptIndex nrows = mtx->nrows;
	sptIndex ncols = mtx->ncols;
	sptIndex stride = mtx->stride;
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4b04:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4b08:	9123a021 	add	x1, x1, #0x8e8
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4b0c:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex nrows = mtx->nrows;
    4b10:	b940001a 	ldr	w26, [x0]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4b14:	aa1503e0 	mov	x0, x21
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4b18:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex ncols = mtx->ncols;
    4b1c:	b94006d9 	ldr	w25, [x22, #4]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4b20:	2a1a03e2 	mov	w2, w26
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4b24:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex stride = mtx->stride;
    4b28:	b9400edb 	ldr	w27, [x22, #12]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4b2c:	2a1903e3 	mov	w3, w25
    4b30:	97fff24c 	bl	1460 <fprintf@plt>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4b34:	37f80800 	tbnz	w0, #31, 4c34 <sptDumpMatrix+0x144>
	for(sptIndex i=0; i < nrows; ++i) {
    4b38:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex ncols = mtx->ncols;
    4b3c:	2a1903f4 	mov	w20, w25
    4b40:	b0000017 	adrp	x23, 5000 <_IO_stdin_used+0x30>
		for(sptIndex j=0; j < ncols; ++j) {
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4b44:	9124c2f7 	add	x23, x23, #0x930
	for(sptIndex i=0; i < nrows; ++i) {
    4b48:	52800018 	mov	w24, #0x0                   	// #0
    4b4c:	340005fa 	cbz	w26, 4c08 <sptDumpMatrix+0x118>
		for(sptIndex j=0; j < ncols; ++j) {
    4b50:	4b190293 	sub	w19, w20, w25
    4b54:	35000099 	cbnz	w25, 4b64 <sptDumpMatrix+0x74>
    4b58:	14000025 	b	4bec <sptDumpMatrix+0xfc>
    4b5c:	6b13029f 	cmp	w20, w19
    4b60:	54000460 	b.eq	4bec <sptDumpMatrix+0xfc>  // b.none
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4b64:	f9400ac2 	ldr	x2, [x22, #16]
    4b68:	aa1703e1 	mov	x1, x23
    4b6c:	aa1503e0 	mov	x0, x21
    4b70:	bc735840 	ldr	s0, [x2, w19, uxtw #2]
		for(sptIndex j=0; j < ncols; ++j) {
    4b74:	11000673 	add	w19, w19, #0x1
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4b78:	1e22c000 	fcvt	d0, s0
    4b7c:	97fff239 	bl	1460 <fprintf@plt>
			spt_CheckOSError(iores < 0, "Mtx Dump");
    4b80:	36fffee0 	tbz	w0, #31, 4b5c <sptDumpMatrix+0x6c>
    4b84:	97fff22b 	bl	1430 <__errno_location@plt>
    4b88:	b9400001 	ldr	w1, [x0]
    4b8c:	aa0003fc 	mov	x28, x0
    4b90:	3140403f 	cmn	w1, #0x10, lsl #12
    4b94:	54fffe40 	b.eq	4b5c <sptDumpMatrix+0x6c>  // b.none
    4b98:	2a0103e0 	mov	w0, w1
    4b9c:	b9006fe1 	str	w1, [sp, #108]
    4ba0:	97fff1e4 	bl	1330 <strerror@plt>
    4ba4:	b9406fe1 	ldr	w1, [sp, #108]
    4ba8:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    4bac:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4bb0:	aa0003e4 	mov	x4, x0
    4bb4:	9123e042 	add	x2, x2, #0x8f8
    4bb8:	91248060 	add	x0, x3, #0x920
    4bbc:	528005c3 	mov	w3, #0x2e                  	// #46
    4bc0:	11404021 	add	w1, w1, #0x10, lsl #12
    4bc4:	9400002f 	bl	4c80 <spt_ComplainError>
    4bc8:	b9400380 	ldr	w0, [x28]
    4bcc:	a94363f7 	ldp	x23, x24, [sp, #48]
    4bd0:	11404000 	add	w0, w0, #0x10, lsl #12
		}
		iores = fprintf(fp, "\n");
	}
	iores = fprintf(fp, "\n");
	return 0;
}
    4bd4:	a94153f3 	ldp	x19, x20, [sp, #16]
    4bd8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4bdc:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4be0:	a94573fb 	ldp	x27, x28, [sp, #80]
    4be4:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4be8:	d65f03c0 	ret
		iores = fprintf(fp, "\n");
    4bec:	aa1503e1 	mov	x1, x21
    4bf0:	52800140 	mov	w0, #0xa                   	// #10
	for(sptIndex i=0; i < nrows; ++i) {
    4bf4:	11000718 	add	w24, w24, #0x1
    4bf8:	0b1b0294 	add	w20, w20, w27
		iores = fprintf(fp, "\n");
    4bfc:	97fff181 	bl	1200 <fputc@plt>
	for(sptIndex i=0; i < nrows; ++i) {
    4c00:	6b18035f 	cmp	w26, w24
    4c04:	54fffa61 	b.ne	4b50 <sptDumpMatrix+0x60>  // b.any
	iores = fprintf(fp, "\n");
    4c08:	aa1503e1 	mov	x1, x21
    4c0c:	52800140 	mov	w0, #0xa                   	// #10
    4c10:	97fff17c 	bl	1200 <fputc@plt>
	return 0;
    4c14:	52800000 	mov	w0, #0x0                   	// #0
}
    4c18:	a94153f3 	ldp	x19, x20, [sp, #16]
    4c1c:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    4c20:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    4c24:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4c28:	a94573fb 	ldp	x27, x28, [sp, #80]
    4c2c:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4c30:	d65f03c0 	ret
	spt_CheckOSError(iores < 0, "Mtx Dump");
    4c34:	97fff1ff 	bl	1430 <__errno_location@plt>
    4c38:	b9400014 	ldr	w20, [x0]
    4c3c:	aa0003f3 	mov	x19, x0
    4c40:	3140429f 	cmn	w20, #0x10, lsl #12
    4c44:	54fff7a0 	b.eq	4b38 <sptDumpMatrix+0x48>  // b.none
    4c48:	2a1403e0 	mov	w0, w20
    4c4c:	97fff1b9 	bl	1330 <strerror@plt>
    4c50:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x30>
    4c54:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x30>
    4c58:	aa0003e4 	mov	x4, x0
    4c5c:	11404281 	add	w1, w20, #0x10, lsl #12
    4c60:	91248060 	add	x0, x3, #0x920
    4c64:	9123e042 	add	x2, x2, #0x8f8
    4c68:	52800543 	mov	w3, #0x2a                  	// #42
    4c6c:	94000005 	bl	4c80 <spt_ComplainError>
    4c70:	b9400260 	ldr	w0, [x19]
    4c74:	11404000 	add	w0, w0, #0x10, lsl #12
    4c78:	17ffffd7 	b	4bd4 <sptDumpMatrix+0xe4>
    4c7c:	d503201f 	nop

0000000000004c80 <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4c80:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4c84:	910003fd 	mov	x29, sp
    4c88:	a90153f3 	stp	x19, x20, [sp, #16]
	g_last_error.errcode = errcode;
    4c8c:	d53bd053 	mrs	x19, tpidr_el0
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4c90:	aa0003f4 	mov	x20, x0
	g_last_error.errcode = errcode;
    4c94:	91400273 	add	x19, x19, #0x0, lsl #12
    4c98:	91004273 	add	x19, x19, #0x10
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4c9c:	a9025bf5 	stp	x21, x22, [sp, #32]
    4ca0:	2a0103f5 	mov	w21, w1
    4ca4:	aa0203f6 	mov	x22, x2
    4ca8:	a90363f7 	stp	x23, x24, [sp, #48]
    4cac:	2a0303f7 	mov	w23, w3
	g_last_error.module = module;
    4cb0:	f9000260 	str	x0, [x19]
	g_last_error.errcode = errcode;
    4cb4:	b9000a61 	str	w1, [x19, #8]
	g_last_error.file = file;
    4cb8:	f9000a62 	str	x2, [x19, #16]
	g_last_error.line = line;
    4cbc:	b9001a63 	str	w3, [x19, #24]
	if(reason) {
		free(g_last_error.reason);
    4cc0:	f9401266 	ldr	x6, [x19, #32]
	if(reason) {
    4cc4:	b4000324 	cbz	x4, 4d28 <spt_ComplainError+0xa8>
		free(g_last_error.reason);
    4cc8:	aa0603e0 	mov	x0, x6
    4ccc:	aa0403f8 	mov	x24, x4
    4cd0:	97fff1bc 	bl	13c0 <free@plt>
		g_last_error.reason = strdup(reason);
    4cd4:	aa1803e0 	mov	x0, x24
    4cd8:	97fff18e 	bl	1310 <strdup@plt>
    4cdc:	f9001260 	str	x0, [x19, #32]
    4ce0:	aa0003e6 	mov	x6, x0
		if(!g_last_error.reason) {
    4ce4:	b4000440 	cbz	x0, 4d6c <spt_ComplainError+0xec>
			abort();
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4ce8:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4cec:	394000c1 	ldrb	w1, [x6]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4cf0:	f947d800 	ldr	x0, [x0, #4016]
    4cf4:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4cf8:	35000241 	cbnz	w1, 4d40 <spt_ComplainError+0xc0>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4cfc:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4d00:	2a1703e5 	mov	w5, w23
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    4d04:	a94363f7 	ldp	x23, x24, [sp, #48]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4d08:	aa1603e4 	mov	x4, x22
    4d0c:	2a1503e3 	mov	w3, w21
    4d10:	aa1403e2 	mov	x2, x20
    4d14:	91256021 	add	x1, x1, #0x958
}
    4d18:	a94153f3 	ldp	x19, x20, [sp, #16]
    4d1c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4d20:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4d24:	17fff1cf 	b	1460 <fprintf@plt>
    4d28:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    4d2c:	f947d800 	ldr	x0, [x0, #4016]
    4d30:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4d34:	b4fffe46 	cbz	x6, 4cfc <spt_ComplainError+0x7c>
    4d38:	394000c1 	ldrb	w1, [x6]
    4d3c:	34fffe01 	cbz	w1, 4cfc <spt_ComplainError+0x7c>
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4d40:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4d44:	2a1703e5 	mov	w5, w23
}
    4d48:	a94363f7 	ldp	x23, x24, [sp, #48]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4d4c:	aa1603e4 	mov	x4, x22
    4d50:	2a1503e3 	mov	w3, w21
    4d54:	aa1403e2 	mov	x2, x20
    4d58:	9124e021 	add	x1, x1, #0x938
}
    4d5c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4d60:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4d64:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4d68:	17fff1be 	b	1460 <fprintf@plt>
			abort();
    4d6c:	97fff181 	bl	1370 <abort@plt>

0000000000004d70 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    4d70:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    4d74:	910003fd 	mov	x29, sp
	double size = (double)bytes;
	int suff = 0;
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
	while(size > 1024 && suff < 5) {
    4d78:	d2e81201 	mov	x1, #0x4090000000000000    	// #4652218415073722368
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4d7c:	9100a3e2 	add	x2, sp, #0x28
{
    4d80:	fd000be8 	str	d8, [sp, #16]
    4d84:	9e670008 	fmov	d8, x0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4d88:	f0000080 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    4d8c:	910a0000 	add	x0, x0, #0x280
	while(size > 1024 && suff < 5) {
    4d90:	9e670021 	fmov	d1, x1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4d94:	a9401c06 	ldp	x6, x7, [x0]
    4d98:	a9411404 	ldp	x4, x5, [x0, #16]
    4d9c:	a9029fe6 	stp	x6, x7, [sp, #40]
	double size = (double)bytes;
    4da0:	7e61d900 	ucvtf	d0, d8
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4da4:	f9401000 	ldr	x0, [x0, #32]
    4da8:	a9011444 	stp	x4, x5, [x2, #16]
	while(size > 1024 && suff < 5) {
    4dac:	1e612010 	fcmpe	d0, d1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4db0:	f9001040 	str	x0, [x2, #32]
	while(size > 1024 && suff < 5) {
    4db4:	54000669 	b.ls	4e80 <sptBytesString+0x110>  // b.plast
    4db8:	d2e82601 	mov	x1, #0x4130000000000000    	// #4697254411347427328
		size /= 1024.;
    4dbc:	d2e7ea00 	mov	x0, #0x3f50000000000000    	// #4562146422526312448
	while(size > 1024 && suff < 5) {
    4dc0:	9e670022 	fmov	d2, x1
		size /= 1024.;
    4dc4:	9e670001 	fmov	d1, x0
	while(size > 1024 && suff < 5) {
    4dc8:	1e622010 	fcmpe	d0, d2
		size /= 1024.;
    4dcc:	1e610802 	fmul	d2, d0, d1
	while(size > 1024 && suff < 5) {
    4dd0:	540005c9 	b.ls	4e88 <sptBytesString+0x118>  // b.plast
    4dd4:	d2e83a00 	mov	x0, #0x41d0000000000000    	// #4742290407621132288
		size /= 1024.;
    4dd8:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4ddc:	9e670003 	fmov	d3, x0
    4de0:	1e632010 	fcmpe	d0, d3
    4de4:	54000589 	b.ls	4e94 <sptBytesString+0x124>  // b.plast
    4de8:	d2e84e00 	mov	x0, #0x4270000000000000    	// #4787326403894837248
		size /= 1024.;
    4dec:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4df0:	9e670003 	fmov	d3, x0
    4df4:	1e632010 	fcmpe	d0, d3
    4df8:	540003e9 	b.ls	4e74 <sptBytesString+0x104>  // b.plast
    4dfc:	d2e86200 	mov	x0, #0x4310000000000000    	// #4832362400168542208
		size /= 1024.;
    4e00:	1e610842 	fmul	d2, d2, d1
	while(size > 1024 && suff < 5) {
    4e04:	9e670003 	fmov	d3, x0
    4e08:	1e632010 	fcmpe	d0, d3
    4e0c:	540004a9 	b.ls	4ea0 <sptBytesString+0x130>  // b.plast
		size /= 1024.;
    4e10:	1e610840 	fmul	d0, d2, d1
		++suff;
    4e14:	528000a3 	mov	w3, #0x5                   	// #5
	}
	char * ret = NULL;
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4e18:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4e1c:	910083e0 	add	x0, sp, #0x20
    4e20:	f863d842 	ldr	x2, [x2, w3, sxtw #3]
    4e24:	91268021 	add	x1, x1, #0x9a0
	char * ret = NULL;
    4e28:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4e2c:	97fff0fd 	bl	1220 <asprintf@plt>
    4e30:	3100041f 	cmn	w0, #0x1
    4e34:	540000a0 	b.eq	4e48 <sptBytesString+0xd8>  // b.none
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
		ret = NULL;
	}
	return ret;
}
    4e38:	fd400be8 	ldr	d8, [sp, #16]
	return ret;
    4e3c:	f94013e0 	ldr	x0, [sp, #32]
}
    4e40:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4e44:	d65f03c0 	ret
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    4e48:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xf89c>
    4e4c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x30>
    4e50:	f947d800 	ldr	x0, [x0, #4016]
    4e54:	9126a021 	add	x1, x1, #0x9a8
    4e58:	9e660102 	fmov	x2, d8
    4e5c:	f9400000 	ldr	x0, [x0]
    4e60:	97fff180 	bl	1460 <fprintf@plt>
    4e64:	d2800000 	mov	x0, #0x0                   	// #0
}
    4e68:	fd400be8 	ldr	d8, [sp, #16]
    4e6c:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4e70:	d65f03c0 	ret
		size /= 1024.;
    4e74:	1e604040 	fmov	d0, d2
		++suff;
    4e78:	52800063 	mov	w3, #0x3                   	// #3
    4e7c:	17ffffe7 	b	4e18 <sptBytesString+0xa8>
	int suff = 0;
    4e80:	52800003 	mov	w3, #0x0                   	// #0
    4e84:	17ffffe5 	b	4e18 <sptBytesString+0xa8>
		size /= 1024.;
    4e88:	1e604040 	fmov	d0, d2
		++suff;
    4e8c:	52800023 	mov	w3, #0x1                   	// #1
    4e90:	17ffffe2 	b	4e18 <sptBytesString+0xa8>
		size /= 1024.;
    4e94:	1e604040 	fmov	d0, d2
		++suff;
    4e98:	52800043 	mov	w3, #0x2                   	// #2
    4e9c:	17ffffdf 	b	4e18 <sptBytesString+0xa8>
		size /= 1024.;
    4ea0:	1e604040 	fmov	d0, d2
		++suff;
    4ea4:	52800083 	mov	w3, #0x4                   	// #4
    4ea8:	17ffffdc 	b	4e18 <sptBytesString+0xa8>
    4eac:	d503201f 	nop

0000000000004eb0 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    4eb0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4eb4:	910003fd 	mov	x29, sp
    4eb8:	fd000be8 	str	d8, [sp, #16]
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    4ebc:	97fff111 	bl	1300 <rand@plt>
    4ec0:	52a61801 	mov	w1, #0x30c00000            	// #817889280
    4ec4:	1e220008 	scvtf	s8, w0
    4ec8:	1e270020 	fmov	s0, w1
    4ecc:	1e200908 	fmul	s8, s8, s0
	if(rand() % 2 == 0) {
    4ed0:	97fff10c 	bl	1300 <rand@plt>
		v *= -1;
    4ed4:	1e214100 	fneg	s0, s8
    4ed8:	f240001f 	tst	x0, #0x1
	}
	return v;
}
    4edc:	1e280c00 	fcsel	s0, s0, s8, eq  // eq = none
    4ee0:	fd400be8 	ldr	d8, [sp, #16]
    4ee4:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4ee8:	d65f03c0 	ret
    4eec:	d503201f 	nop

0000000000004ef0 <__aarch64_cas4_acq_rel>:
    4ef0:	d503245f 	bti	c
    4ef4:	f0000090 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    4ef8:	394aa610 	ldrb	w16, [x16, #681]
    4efc:	34000070 	cbz	w16, 4f08 <__aarch64_cas4_acq_rel+0x18>
    4f00:	88e0fc41 	casal	w0, w1, [x2]
    4f04:	d65f03c0 	ret
    4f08:	2a0003f0 	mov	w16, w0
    4f0c:	885ffc40 	ldaxr	w0, [x2]
    4f10:	6b10001f 	cmp	w0, w16
    4f14:	54000061 	b.ne	4f20 <__aarch64_cas4_acq_rel+0x30>  // b.any
    4f18:	8811fc41 	stlxr	w17, w1, [x2]
    4f1c:	35ffff91 	cbnz	w17, 4f0c <__aarch64_cas4_acq_rel+0x1c>
    4f20:	d65f03c0 	ret
    4f24:	d503201f 	nop
    4f28:	d503201f 	nop
    4f2c:	d503201f 	nop

0000000000004f30 <__libc_csu_init>:
    4f30:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4f34:	910003fd 	mov	x29, sp
    4f38:	a90153f3 	stp	x19, x20, [sp, #16]
    4f3c:	d0000094 	adrp	x20, 16000 <__FRAME_END__+0xf89c>
    4f40:	9134e294 	add	x20, x20, #0xd38
    4f44:	a9025bf5 	stp	x21, x22, [sp, #32]
    4f48:	d0000095 	adrp	x21, 16000 <__FRAME_END__+0xf89c>
    4f4c:	913482b5 	add	x21, x21, #0xd20
    4f50:	cb150294 	sub	x20, x20, x21
    4f54:	2a0003f6 	mov	w22, w0
    4f58:	a90363f7 	stp	x23, x24, [sp, #48]
    4f5c:	aa0103f7 	mov	x23, x1
    4f60:	aa0203f8 	mov	x24, x2
    4f64:	9343fe94 	asr	x20, x20, #3
    4f68:	97fff08a 	bl	1190 <_init>
    4f6c:	b4000174 	cbz	x20, 4f98 <__libc_csu_init+0x68>
    4f70:	d2800013 	mov	x19, #0x0                   	// #0
    4f74:	d503201f 	nop
    4f78:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    4f7c:	aa1803e2 	mov	x2, x24
    4f80:	91000673 	add	x19, x19, #0x1
    4f84:	aa1703e1 	mov	x1, x23
    4f88:	2a1603e0 	mov	w0, w22
    4f8c:	d63f0060 	blr	x3
    4f90:	eb13029f 	cmp	x20, x19
    4f94:	54ffff21 	b.ne	4f78 <__libc_csu_init+0x48>  // b.any
    4f98:	a94153f3 	ldp	x19, x20, [sp, #16]
    4f9c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4fa0:	a94363f7 	ldp	x23, x24, [sp, #48]
    4fa4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    4fa8:	d65f03c0 	ret
    4fac:	d503201f 	nop

0000000000004fb0 <__libc_csu_fini>:
    4fb0:	d65f03c0 	ret

Disassembly of section .fini:

0000000000004fb4 <_fini>:
    4fb4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    4fb8:	910003fd 	mov	x29, sp
    4fbc:	a8c17bfd 	ldp	x29, x30, [sp], #16
    4fc0:	d65f03c0 	ret
