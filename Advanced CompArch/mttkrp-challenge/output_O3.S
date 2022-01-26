
mttkrp:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000001178 <_init>:
    1178:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    117c:	910003fd 	mov	x29, sp
    1180:	9400028f 	bl	1bbc <call_weak_fn>
    1184:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1188:	d65f03c0 	ret

Disassembly of section .plt:

0000000000001190 <.plt>:
    1190:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
    1194:	b00000b0 	adrp	x16, 16000 <__FRAME_END__+0xfb88>
    1198:	f947fe11 	ldr	x17, [x16, #4088]
    119c:	913fe210 	add	x16, x16, #0xff8
    11a0:	d61f0220 	br	x17
    11a4:	d503201f 	nop
    11a8:	d503201f 	nop
    11ac:	d503201f 	nop

00000000000011b0 <memcpy@plt>:
    11b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11b4:	f9400211 	ldr	x17, [x16]
    11b8:	91000210 	add	x16, x16, #0x0
    11bc:	d61f0220 	br	x17

00000000000011c0 <exit@plt>:
    11c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11c4:	f9400611 	ldr	x17, [x16, #8]
    11c8:	91002210 	add	x16, x16, #0x8
    11cc:	d61f0220 	br	x17

00000000000011d0 <__cxa_finalize@plt>:
    11d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11d4:	f9400a11 	ldr	x17, [x16, #16]
    11d8:	91004210 	add	x16, x16, #0x10
    11dc:	d61f0220 	br	x17

00000000000011e0 <fputc@plt>:
    11e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11e4:	f9400e11 	ldr	x17, [x16, #24]
    11e8:	91006210 	add	x16, x16, #0x18
    11ec:	d61f0220 	br	x17

00000000000011f0 <clock_gettime@plt>:
    11f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    11f4:	f9401211 	ldr	x17, [x16, #32]
    11f8:	91008210 	add	x16, x16, #0x20
    11fc:	d61f0220 	br	x17

0000000000001200 <asprintf@plt>:
    1200:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1204:	f9401611 	ldr	x17, [x16, #40]
    1208:	9100a210 	add	x16, x16, #0x28
    120c:	d61f0220 	br	x17

0000000000001210 <pow@plt>:
    1210:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1214:	f9401a11 	ldr	x17, [x16, #48]
    1218:	9100c210 	add	x16, x16, #0x30
    121c:	d61f0220 	br	x17

0000000000001220 <omp_get_thread_num@plt>:
    1220:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1224:	f9401e11 	ldr	x17, [x16, #56]
    1228:	9100e210 	add	x16, x16, #0x38
    122c:	d61f0220 	br	x17

0000000000001230 <fclose@plt>:
    1230:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1234:	f9402211 	ldr	x17, [x16, #64]
    1238:	91010210 	add	x16, x16, #0x40
    123c:	d61f0220 	br	x17

0000000000001240 <fopen@plt>:
    1240:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1244:	f9402611 	ldr	x17, [x16, #72]
    1248:	91012210 	add	x16, x16, #0x48
    124c:	d61f0220 	br	x17

0000000000001250 <time@plt>:
    1250:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1254:	f9402a11 	ldr	x17, [x16, #80]
    1258:	91014210 	add	x16, x16, #0x50
    125c:	d61f0220 	br	x17

0000000000001260 <malloc@plt>:
    1260:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1264:	f9402e11 	ldr	x17, [x16, #88]
    1268:	91016210 	add	x16, x16, #0x58
    126c:	d61f0220 	br	x17

0000000000001270 <__isoc99_fscanf@plt>:
    1270:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1274:	f9403211 	ldr	x17, [x16, #96]
    1278:	91018210 	add	x16, x16, #0x60
    127c:	d61f0220 	br	x17

0000000000001280 <__libc_start_main@plt>:
    1280:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1284:	f9403611 	ldr	x17, [x16, #104]
    1288:	9101a210 	add	x16, x16, #0x68
    128c:	d61f0220 	br	x17

0000000000001290 <fgetc@plt>:
    1290:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1294:	f9403a11 	ldr	x17, [x16, #112]
    1298:	9101c210 	add	x16, x16, #0x70
    129c:	d61f0220 	br	x17

00000000000012a0 <aligned_alloc@plt>:
    12a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12a4:	f9403e11 	ldr	x17, [x16, #120]
    12a8:	9101e210 	add	x16, x16, #0x78
    12ac:	d61f0220 	br	x17

00000000000012b0 <memset@plt>:
    12b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12b4:	f9404211 	ldr	x17, [x16, #128]
    12b8:	91020210 	add	x16, x16, #0x80
    12bc:	d61f0220 	br	x17

00000000000012c0 <calloc@plt>:
    12c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12c4:	f9404611 	ldr	x17, [x16, #136]
    12c8:	91022210 	add	x16, x16, #0x88
    12cc:	d61f0220 	br	x17

00000000000012d0 <realloc@plt>:
    12d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12d4:	f9404a11 	ldr	x17, [x16, #144]
    12d8:	91024210 	add	x16, x16, #0x90
    12dc:	d61f0220 	br	x17

00000000000012e0 <rand@plt>:
    12e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12e4:	f9404e11 	ldr	x17, [x16, #152]
    12e8:	91026210 	add	x16, x16, #0x98
    12ec:	d61f0220 	br	x17

00000000000012f0 <strdup@plt>:
    12f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    12f4:	f9405211 	ldr	x17, [x16, #160]
    12f8:	91028210 	add	x16, x16, #0xa0
    12fc:	d61f0220 	br	x17

0000000000001300 <omp_get_num_threads@plt>:
    1300:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1304:	f9405611 	ldr	x17, [x16, #168]
    1308:	9102a210 	add	x16, x16, #0xa8
    130c:	d61f0220 	br	x17

0000000000001310 <strerror@plt>:
    1310:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1314:	f9405a11 	ldr	x17, [x16, #176]
    1318:	9102c210 	add	x16, x16, #0xb0
    131c:	d61f0220 	br	x17

0000000000001320 <strrchr@plt>:
    1320:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1324:	f9405e11 	ldr	x17, [x16, #184]
    1328:	9102e210 	add	x16, x16, #0xb8
    132c:	d61f0220 	br	x17

0000000000001330 <__gmon_start__@plt>:
    1330:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1334:	f9406211 	ldr	x17, [x16, #192]
    1338:	91030210 	add	x16, x16, #0xc0
    133c:	d61f0220 	br	x17

0000000000001340 <__getauxval@plt>:
    1340:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1344:	f9406611 	ldr	x17, [x16, #200]
    1348:	91032210 	add	x16, x16, #0xc8
    134c:	d61f0220 	br	x17

0000000000001350 <abort@plt>:
    1350:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1354:	f9406a11 	ldr	x17, [x16, #208]
    1358:	91034210 	add	x16, x16, #0xd0
    135c:	d61f0220 	br	x17

0000000000001360 <puts@plt>:
    1360:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1364:	f9406e11 	ldr	x17, [x16, #216]
    1368:	91036210 	add	x16, x16, #0xd8
    136c:	d61f0220 	br	x17

0000000000001370 <getopt_long@plt>:
    1370:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1374:	f9407211 	ldr	x17, [x16, #224]
    1378:	91038210 	add	x16, x16, #0xe0
    137c:	d61f0220 	br	x17

0000000000001380 <strcmp@plt>:
    1380:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1384:	f9407611 	ldr	x17, [x16, #232]
    1388:	9103a210 	add	x16, x16, #0xe8
    138c:	d61f0220 	br	x17

0000000000001390 <fread@plt>:
    1390:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1394:	f9407a11 	ldr	x17, [x16, #240]
    1398:	9103c210 	add	x16, x16, #0xf0
    139c:	d61f0220 	br	x17

00000000000013a0 <free@plt>:
    13a0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13a4:	f9407e11 	ldr	x17, [x16, #248]
    13a8:	9103e210 	add	x16, x16, #0xf8
    13ac:	d61f0220 	br	x17

00000000000013b0 <fwrite@plt>:
    13b0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13b4:	f9408211 	ldr	x17, [x16, #256]
    13b8:	91040210 	add	x16, x16, #0x100
    13bc:	d61f0220 	br	x17

00000000000013c0 <fflush@plt>:
    13c0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13c4:	f9408611 	ldr	x17, [x16, #264]
    13c8:	91042210 	add	x16, x16, #0x108
    13cc:	d61f0220 	br	x17

00000000000013d0 <strcpy@plt>:
    13d0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13d4:	f9408a11 	ldr	x17, [x16, #272]
    13d8:	91044210 	add	x16, x16, #0x110
    13dc:	d61f0220 	br	x17

00000000000013e0 <srand@plt>:
    13e0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13e4:	f9408e11 	ldr	x17, [x16, #280]
    13e8:	91046210 	add	x16, x16, #0x118
    13ec:	d61f0220 	br	x17

00000000000013f0 <__isoc99_sscanf@plt>:
    13f0:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    13f4:	f9409211 	ldr	x17, [x16, #288]
    13f8:	91048210 	add	x16, x16, #0x120
    13fc:	d61f0220 	br	x17

0000000000001400 <printf@plt>:
    1400:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1404:	f9409611 	ldr	x17, [x16, #296]
    1408:	9104a210 	add	x16, x16, #0x128
    140c:	d61f0220 	br	x17

0000000000001410 <__errno_location@plt>:
    1410:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1414:	f9409a11 	ldr	x17, [x16, #304]
    1418:	9104c210 	add	x16, x16, #0x130
    141c:	d61f0220 	br	x17

0000000000001420 <GOMP_parallel@plt>:
    1420:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1424:	f9409e11 	ldr	x17, [x16, #312]
    1428:	9104e210 	add	x16, x16, #0x138
    142c:	d61f0220 	br	x17

0000000000001430 <putchar@plt>:
    1430:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1434:	f940a211 	ldr	x17, [x16, #320]
    1438:	91050210 	add	x16, x16, #0x140
    143c:	d61f0220 	br	x17

0000000000001440 <fprintf@plt>:
    1440:	d00000b0 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    1444:	f940a611 	ldr	x17, [x16, #328]
    1448:	91052210 	add	x16, x16, #0x148
    144c:	d61f0220 	br	x17

Disassembly of section .text:

0000000000001450 <main>:

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in COO format, matrices are dense.
 */
int main(int argc, char ** argv)
{
    1450:	d13203ff 	sub	sp, sp, #0xc80
	sptSparseTensor X;
	sptMatrix ** U;

	bool random = true;
	sptIndex mode = 0;
	sptIndex R = 16;
    1454:	52800203 	mov	w3, #0x10                  	// #16
	int dev_id = -2;
    1458:	12800022 	mov	w2, #0xfffffffe            	// #-2
{
    145c:	a9007bfd 	stp	x29, x30, [sp]
    1460:	910003fd 	mov	x29, sp
    1464:	a90153f3 	stp	x19, x20, [sp, #16]
    1468:	2a0003f3 	mov	w19, w0
    146c:	aa0103f4 	mov	x20, x1
	int niters = 5;
	int nthreads = 1;
	printf("niters: %d\n", niters);
    1470:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1474:	528000a1 	mov	w1, #0x5                   	// #5
    1478:	913cc000 	add	x0, x0, #0xf30
{
    147c:	a9025bf5 	stp	x21, x22, [sp, #32]
    1480:	a90363f7 	stp	x23, x24, [sp, #48]
    1484:	a9046bf9 	stp	x25, x26, [sp, #64]
    1488:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex R = 16;
    148c:	290e0fff 	stp	wzr, w3, [sp, #112]
	int dev_id = -2;
    1490:	b9007be2 	str	w2, [sp, #120]
	printf("niters: %d\n", niters);
    1494:	97ffffdb 	bl	1400 <printf@plt>

	if(argc <= 3) { // #Required arguments
    1498:	71000e7f 	cmp	w19, #0x3
    149c:	540035ad 	b.le	1b50 <main+0x700>
			case 'i':
				strcpy(fname, optarg);
				printf("input file: %s\n", fname); fflush(stdout);
				break;
			case 'o':
				fo = fopen(optarg, "w");
    14a0:	b00000ba 	adrp	x26, 16000 <__FRAME_END__+0xfb88>
    14a4:	d00000b7 	adrp	x23, 17000 <memcpy@GLIBC_2.17>
    14a8:	9000003c 	adrp	x28, 5000 <_IO_stdin_used+0x2f0>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14ac:	910582f7 	add	x23, x23, #0x160
				fo = fopen(optarg, "w");
    14b0:	f947df5b 	ldr	x27, [x26, #4024]
    14b4:	9118239c 	add	x28, x28, #0x608
    14b8:	9101f3f9 	add	x25, sp, #0x7c
	bool random = true;
    14bc:	52800035 	mov	w21, #0x1                   	// #1
	FILE *fo = NULL;
    14c0:	d2800016 	mov	x22, #0x0                   	// #0
    14c4:	f0000018 	adrp	x24, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14c8:	aa1903e4 	mov	x4, x25
    14cc:	aa1703e3 	mov	x3, x23
    14d0:	913d0302 	add	x2, x24, #0xf40
    14d4:	aa1403e1 	mov	x1, x20
    14d8:	2a1303e0 	mov	w0, w19
		int option_index = 0;
    14dc:	b9007fff 	str	wzr, [sp, #124]
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14e0:	97ffffa4 	bl	1370 <getopt_long@plt>
		if(c == -1) {
    14e4:	3100041f 	cmn	w0, #0x1
    14e8:	54000ae0 	b.eq	1644 <main+0x1f4>  // b.none
		switch(c) {
    14ec:	7101bc1f 	cmp	w0, #0x6f
    14f0:	54000900 	b.eq	1610 <main+0x1c0>  // b.none
    14f4:	5400018c 	b.gt	1524 <main+0xd4>
    14f8:	7101a41f 	cmp	w0, #0x69
    14fc:	54000760 	b.eq	15e8 <main+0x198>  // b.none
    1500:	7101b41f 	cmp	w0, #0x6d
    1504:	54000221 	b.ne	1548 <main+0xf8>  // b.any
				strcpy(foname, optarg);
				sptAssert(fo != NULL);
				printf("output file: %s\n", optarg); fflush(stdout);
				break;
			case 'm':
				sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
    1508:	f947df40 	ldr	x0, [x26, #4024]
    150c:	9101c3e2 	add	x2, sp, #0x70
    1510:	f0000001 	adrp	x1, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1514:	913de021 	add	x1, x1, #0xf78
    1518:	f9400000 	ldr	x0, [x0]
    151c:	97ffffb5 	bl	13f0 <__isoc99_sscanf@plt>
				break;
    1520:	17ffffea 	b	14c8 <main+0x78>
		switch(c) {
    1524:	7101c81f 	cmp	w0, #0x72
    1528:	540003c1 	b.ne	15a0 <main+0x150>  // b.any
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
					exit(1);
				}
				break;
			case 'r':
				sscanf(optarg, "%u"PASTA_SCN_INDEX, &R);
    152c:	f947df40 	ldr	x0, [x26, #4024]
    1530:	9101d3e2 	add	x2, sp, #0x74
    1534:	f0000001 	adrp	x1, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1538:	913ea021 	add	x1, x1, #0xfa8
    153c:	f9400000 	ldr	x0, [x0]
    1540:	97ffffac 	bl	13f0 <__isoc99_sscanf@plt>
				break;
    1544:	17ffffe1 	b	14c8 <main+0x78>
		switch(c) {
    1548:	7101901f 	cmp	w0, #0x64
    154c:	54003021 	b.ne	1b50 <main+0x700>  // b.any
				sscanf(optarg, "%d", &dev_id);
    1550:	f947df40 	ldr	x0, [x26, #4024]
    1554:	9101e3e2 	add	x2, sp, #0x78
    1558:	f0000001 	adrp	x1, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    155c:	913e0021 	add	x1, x1, #0xf80
    1560:	f9400000 	ldr	x0, [x0]
    1564:	97ffffa3 	bl	13f0 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    1568:	b9407be0 	ldr	w0, [sp, #120]
    156c:	11000800 	add	w0, w0, #0x2
    1570:	7100041f 	cmp	w0, #0x1
    1574:	54fffaa9 	b.ls	14c8 <main+0x78>  // b.plast
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    1578:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xfb88>
    157c:	d2800382 	mov	x2, #0x1c                  	// #28
    1580:	d2800021 	mov	x1, #0x1                   	// #1
    1584:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1588:	f947d863 	ldr	x3, [x3, #4016]
    158c:	913e2000 	add	x0, x0, #0xf88
    1590:	f9400063 	ldr	x3, [x3]
    1594:	97ffff87 	bl	13b0 <fwrite@plt>
					exit(1);
    1598:	52800020 	mov	w0, #0x1                   	// #1
    159c:	97ffff09 	bl	11c0 <exit@plt>
		switch(c) {
    15a0:	7101d81f 	cmp	w0, #0x76
    15a4:	54002d61 	b.ne	1b50 <main+0x700>  // b.any
			case 'v':
				random = false;
				strcpy(fvname, optarg);
    15a8:	f947df41 	ldr	x1, [x26, #4024]
    15ac:	9112c3e2 	add	x2, sp, #0x4b0
    15b0:	aa0203e0 	mov	x0, x2
				random = false;
    15b4:	52800015 	mov	w21, #0x0                   	// #0
				strcpy(fvname, optarg);
    15b8:	f9400021 	ldr	x1, [x1]
    15bc:	97ffff85 	bl	13d0 <strcpy@plt>
    15c0:	aa0003e2 	mov	x2, x0
				printf("validation input file: %s\n", fvname); fflush(stdout);
    15c4:	aa0203e1 	mov	x1, x2
    15c8:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    15cc:	913ec000 	add	x0, x0, #0xfb0
				printf("output file: %s\n", optarg); fflush(stdout);
    15d0:	97ffff8c 	bl	1400 <printf@plt>
    15d4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    15d8:	f947e000 	ldr	x0, [x0, #4032]
    15dc:	f9400000 	ldr	x0, [x0]
    15e0:	97ffff78 	bl	13c0 <fflush@plt>
				break;
    15e4:	17ffffb9 	b	14c8 <main+0x78>
				strcpy(fname, optarg);
    15e8:	f947df41 	ldr	x1, [x26, #4024]
    15ec:	910323e2 	add	x2, sp, #0xc8
    15f0:	aa0203e0 	mov	x0, x2
    15f4:	f9400021 	ldr	x1, [x1]
    15f8:	97ffff76 	bl	13d0 <strcpy@plt>
    15fc:	aa0003e2 	mov	x2, x0
				printf("input file: %s\n", fname); fflush(stdout);
    1600:	aa0203e1 	mov	x1, x2
    1604:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1608:	913d4000 	add	x0, x0, #0xf50
    160c:	17fffff1 	b	15d0 <main+0x180>
				fo = fopen(optarg, "w");
    1610:	f9400360 	ldr	x0, [x27]
    1614:	aa1c03e1 	mov	x1, x28
    1618:	97ffff0a 	bl	1240 <fopen@plt>
    161c:	aa0003f6 	mov	x22, x0
				strcpy(foname, optarg);
    1620:	f9400361 	ldr	x1, [x27]
    1624:	912263e0 	add	x0, sp, #0x898
    1628:	f90037e1 	str	x1, [sp, #104]
    162c:	97ffff69 	bl	13d0 <strcpy@plt>
				sptAssert(fo != NULL);
    1630:	f94037e1 	ldr	x1, [sp, #104]
    1634:	b4001cd6 	cbz	x22, 19cc <main+0x57c>
				printf("output file: %s\n", optarg); fflush(stdout);
    1638:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    163c:	913d8000 	add	x0, x0, #0xf60
    1640:	17ffffe4 	b	15d0 <main+0x180>
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1644:	b94073e1 	ldr	w1, [sp, #112]
    1648:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    164c:	913f4000 	add	x0, x0, #0xfd0
	printf("dev_id: %d\n", dev_id);

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1650:	910223f9 	add	x25, sp, #0x88
	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1654:	97ffff6b 	bl	1400 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    1658:	b9407be1 	ldr	w1, [sp, #120]
    165c:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1660:	913f8000 	add	x0, x0, #0xfe0
    1664:	97ffff67 	bl	1400 <printf@plt>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1668:	aa1903e0 	mov	x0, x25
    166c:	910323e2 	add	x2, sp, #0xc8
    1670:	52800021 	mov	w1, #0x1                   	// #1
    1674:	94000424 	bl	2704 <sptLoadSparseTensor>
    1678:	35001aa0 	cbnz	w0, 19cc <main+0x57c>
	sptSparseTensorStatus(&X, stdout);
    167c:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xfb88>
    1680:	aa1903e0 	mov	x0, x25
    1684:	f947e021 	ldr	x1, [x1, #4032]
    1688:	f9400021 	ldr	x1, [x1]
    168c:	94000769 	bl	3430 <sptSparseTensorStatus>

	sptIndex nmodes = X.nmodes;
    1690:	b9408bf4 	ldr	w20, [sp, #136]
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    1694:	11000680 	add	w0, w20, #0x1
    1698:	1100069a 	add	w26, w20, #0x1
    169c:	d37df000 	lsl	x0, x0, #3
    16a0:	97fffef0 	bl	1260 <malloc@plt>
    16a4:	aa0003f3 	mov	x19, x0
	for(sptIndex m=0; m<nmodes+1; ++m) {
    16a8:	aa0003f7 	mov	x23, x0
    16ac:	8b3a4c18 	add	x24, x0, w26, uxtw #3
    16b0:	3400013a 	cbz	w26, 16d4 <main+0x284>
    16b4:	d503201f 	nop
		U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    16b8:	d2800300 	mov	x0, #0x18                  	// #24
    16bc:	97fffee9 	bl	1260 <malloc@plt>
    16c0:	f80086e0 	str	x0, [x23], #8
	for(sptIndex m=0; m<nmodes+1; ++m) {
    16c4:	eb1802ff 	cmp	x23, x24
    16c8:	54ffff81 	b.ne	16b8 <main+0x268>  // b.any
	}
	sptIndex max_ndims = 0;
    16cc:	52800018 	mov	w24, #0x0                   	// #0
	for(sptIndex m=0; m<nmodes; ++m) {
    16d0:	340002b4 	cbz	w20, 1724 <main+0x2d4>
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    16d4:	f9404fe0 	ldr	x0, [sp, #152]
    16d8:	d2800017 	mov	x23, #0x0                   	// #0
    16dc:	52800018 	mov	w24, #0x0                   	// #0
    16e0:	b8777801 	ldr	w1, [x0, x23, lsl #2]
    16e4:	d37ef6fa 	lsl	x26, x23, #2
    16e8:	b94077e2 	ldr	w2, [sp, #116]
    16ec:	f8777a60 	ldr	x0, [x19, x23, lsl #3]
    16f0:	94000694 	bl	3140 <sptNewMatrix>
    16f4:	350016c0 	cbnz	w0, 19cc <main+0x57c>
		// sptAssert(sptConstantMatrix(U[m], 1) == 0);
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    16f8:	f8777a60 	ldr	x0, [x19, x23, lsl #3]
    16fc:	2a1503e1 	mov	w1, w21
    1700:	940006c0 	bl	3200 <sptRandomizeMatrix>
    1704:	35001640 	cbnz	w0, 19cc <main+0x57c>
		if(X.ndims[m] > max_ndims)
    1708:	f9404fe0 	ldr	x0, [sp, #152]
	for(sptIndex m=0; m<nmodes; ++m) {
    170c:	910006f7 	add	x23, x23, #0x1
    1710:	b87a6801 	ldr	w1, [x0, x26]
    1714:	6b01031f 	cmp	w24, w1
    1718:	1a812318 	csel	w24, w24, w1, cs  // cs = hs, nlast
    171c:	6b17029f 	cmp	w20, w23
    1720:	54fffe08 	b.hi	16e0 <main+0x290>  // b.pmore
			max_ndims = X.ndims[m];
	}
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1724:	2a1403fb 	mov	w27, w20
    1728:	d37d7e80 	ubfiz	x0, x20, #3, #32
    172c:	8b00027a 	add	x26, x19, x0
    1730:	b94077e2 	ldr	w2, [sp, #116]
    1734:	f90037e0 	str	x0, [sp, #104]
    1738:	2a1803e1 	mov	w1, w24
    173c:	f87b7a60 	ldr	x0, [x19, x27, lsl #3]
    1740:	94000680 	bl	3140 <sptNewMatrix>
    1744:	35001440 	cbnz	w0, 19cc <main+0x57c>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1748:	0f000400 	movi	v0.2s, #0x0
    174c:	f87b7a60 	ldr	x0, [x19, x27, lsl #3]
    1750:	940006f4 	bl	3320 <sptConstantMatrix>
    1754:	350013c0 	cbnz	w0, 19cc <main+0x57c>
	sptIndex stride = U[0]->stride;

	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    1758:	d37ef760 	lsl	x0, x27, #2
    175c:	97fffec1 	bl	1260 <malloc@plt>
    1760:	aa0003f8 	mov	x24, x0
	mats_order[0] = mode;
    1764:	b94073e3 	ldr	w3, [sp, #112]
	for(sptIndex i=1; i<nmodes; ++i)
    1768:	d2800020 	mov	x0, #0x1                   	// #1
	mats_order[0] = mode;
    176c:	b9000303 	str	w3, [x24]
	for(sptIndex i=1; i<nmodes; ++i)
    1770:	7100069f 	cmp	w20, #0x1
    1774:	54000109 	b.ls	1794 <main+0x344>  // b.plast
		mats_order[i] = (mode+i) % nmodes;
    1778:	0b000062 	add	w2, w3, w0
    177c:	1ad40841 	udiv	w1, w2, w20
    1780:	1b148821 	msub	w1, w1, w20, w2
    1784:	b8207b01 	str	w1, [x24, x0, lsl #2]
	for(sptIndex i=1; i<nmodes; ++i)
    1788:	91000400 	add	x0, x0, #0x1
    178c:	6b00029f 	cmp	w20, w0
    1790:	54ffff48 	b.hi	1778 <main+0x328>  // b.pmore

	/* For warm-up caches, timing not included */
	if(dev_id == -2) {
    1794:	b9407be0 	ldr	w0, [sp, #120]
    1798:	3100081f 	cmn	w0, #0x2
    179c:	54001780 	b.eq	1a8c <main+0x63c>  // b.none
		nthreads = 1;
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
	} else if(dev_id == -1) {
    17a0:	3100041f 	cmn	w0, #0x1
    17a4:	54001800 	b.eq	1aa4 <main+0x654>  // b.none
    17a8:	910203f7 	add	x23, sp, #0x80
		nthreads = 1;
    17ac:	5280003c 	mov	w28, #0x1                   	// #1
#endif
	}


	sptTimer timer;
	sptNewTimer(&timer, 0);
    17b0:	aa1703e0 	mov	x0, x23
    17b4:	52800001 	mov	w1, #0x0                   	// #0
    17b8:	94000b42 	bl	44c0 <sptNewTimer>
	sptStartTimer(timer);
    17bc:	528000b7 	mov	w23, #0x5                   	// #5
    17c0:	f94043e0 	ldr	x0, [sp, #128]
    17c4:	94000b5f 	bl	4540 <sptStartTimer>

	for(int it=0; it<niters; ++it) {
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    17c8:	0f000400 	movi	v0.2s, #0x0
    17cc:	f9400340 	ldr	x0, [x26]
    17d0:	940006d4 	bl	3320 <sptConstantMatrix>
    17d4:	35000fc0 	cbnz	w0, 19cc <main+0x57c>
		if(dev_id == -2) {
    17d8:	b9407be0 	ldr	w0, [sp, #120]
    17dc:	3100081f 	cmn	w0, #0x2
    17e0:	54000ea0 	b.eq	19b4 <main+0x564>  // b.none
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
		} else if(dev_id == -1) {
    17e4:	3100041f 	cmn	w0, #0x1
    17e8:	54000f60 	b.eq	19d4 <main+0x584>  // b.none
	for(int it=0; it<niters; ++it) {
    17ec:	710006f7 	subs	w23, w23, #0x1
    17f0:	54fffec1 	b.ne	17c8 <main+0x378>  // b.any
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
#endif
		}
	}

	sptStopTimer(timer);
    17f4:	f94043e0 	ldr	x0, [sp, #128]
    17f8:	94000b82 	bl	4600 <sptStopTimer>

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    17fc:	f94043e0 	ldr	x0, [sp, #128]
    1800:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    1804:	528000a1 	mov	w1, #0x5                   	// #5
    1808:	91000042 	add	x2, x2, #0x0
    180c:	94000bda 	bl	4774 <sptPrintAverageElapsedTime>
    1810:	1e604006 	fmov	d6, d0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1814:	b94077e5 	ldr	w5, [sp, #116]
    1818:	1e630282 	ucvtf	d2, w20
    181c:	f94053e0 	ldr	x0, [sp, #160]
    1820:	d2d9aca2 	mov	x2, #0xcd6500000000        	// #225833675390976
    1824:	1e6300a3 	ucvtf	d3, w5
    1828:	f2e839a2 	movk	x2, #0x41cd, lsl #48
    182c:	9e630000 	ucvtf	d0, x0
    1830:	9e670041 	fmov	d1, x2
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1834:	9b000360 	madd	x0, x27, x0, x0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1838:	1e630842 	fmul	d2, d2, d3
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    183c:	d37ef400 	lsl	x0, x0, #2
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1840:	1e600842 	fmul	d2, d2, d0
    1844:	1e661842 	fdiv	d2, d2, d6
    1848:	1e611840 	fdiv	d0, d2, d1
	for (sptIndex m=0; m<nmodes; ++m) {
    184c:	34001594 	cbz	w20, 1afc <main+0x6ac>
    1850:	51000681 	sub	w1, w20, #0x1
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1854:	f9404fe4 	ldr	x4, [sp, #152]
    1858:	71000c3f 	cmp	w1, #0x3
    185c:	540002c9 	b.ls	18b4 <main+0x464>  // b.plast
    1860:	53027e83 	lsr	w3, w20, #2
    1864:	91004082 	add	x2, x4, #0x10
    1868:	51000463 	sub	w3, w3, #0x1
    186c:	4e040ca5 	dup	v5.4s, w5
    1870:	4f000403 	movi	v3.4s, #0x0
    1874:	aa0403e1 	mov	x1, x4
    1878:	4f000484 	movi	v4.4s, #0x4
    187c:	8b235042 	add	x2, x2, w3, uxtw #4
    1880:	3cc10421 	ldr	q1, [x1], #16
    1884:	4ea59c21 	mul	v1.4s, v1.4s, v5.4s
    1888:	2ea4c022 	umull	v2.2d, v1.2s, v4.2s
    188c:	6ea48022 	umlal2	v2.2d, v1.4s, v4.4s
    1890:	4ee28463 	add	v3.2d, v3.2d, v2.2d
	for (sptIndex m=0; m<nmodes; ++m) {
    1894:	eb01005f 	cmp	x2, x1
    1898:	54ffff41 	b.ne	1880 <main+0x430>  // b.any
    189c:	5ef1b863 	addp	d3, v3.2d
    18a0:	121e7697 	and	w23, w20, #0xfffffffc
    18a4:	4e083c61 	mov	x1, v3.d[0]
    18a8:	8b010000 	add	x0, x0, x1
    18ac:	f240069f 	tst	x20, #0x3
    18b0:	540002c0 	b.eq	1908 <main+0x4b8>  // b.none
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18b4:	b8775881 	ldr	w1, [x4, w23, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18b8:	110006e2 	add	w2, w23, #0x1
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18bc:	1b017ca1 	mul	w1, w5, w1
    18c0:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    18c4:	6b14005f 	cmp	w2, w20
    18c8:	54000202 	b.cs	1908 <main+0x4b8>  // b.hs, b.nlast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18cc:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18d0:	11000ae2 	add	w2, w23, #0x2
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18d4:	1b017ca1 	mul	w1, w5, w1
    18d8:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    18dc:	6b14005f 	cmp	w2, w20
    18e0:	54000142 	b.cs	1908 <main+0x4b8>  // b.hs, b.nlast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18e4:	b8625881 	ldr	w1, [x4, w2, uxtw #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    18e8:	11000ef7 	add	w23, w23, #0x3
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18ec:	1b017ca1 	mul	w1, w5, w1
    18f0:	8b214800 	add	x0, x0, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    18f4:	6b17029f 	cmp	w20, w23
    18f8:	54000089 	b.ls	1908 <main+0x4b8>  // b.plast
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18fc:	b8775881 	ldr	w1, [x4, w23, uxtw #2]
    1900:	1b017ca1 	mul	w1, w5, w1
    1904:	8b214800 	add	x0, x0, w1, uxtw #2
	}
	double gbw = (double)bytes / aver_time / 1e9;
    1908:	9e630002 	ucvtf	d2, x0
    190c:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1910:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    1914:	9e670001 	fmov	d1, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1918:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    191c:	91034000 	add	x0, x0, #0xd0
	double gbw = (double)bytes / aver_time / 1e9;
    1920:	1e661842 	fdiv	d2, d2, d6
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1924:	1e611841 	fdiv	d1, d2, d1
    1928:	97fffeb6 	bl	1400 <printf@plt>

	if(fo != NULL) {
    192c:	b4000116 	cbz	x22, 194c <main+0x4fc>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1930:	f94037e0 	ldr	x0, [sp, #104]
    1934:	aa1603e1 	mov	x1, x22
    1938:	f8606a60 	ldr	x0, [x19, x0]
    193c:	94000bc9 	bl	4860 <sptDumpMatrix>
    1940:	35000460 	cbnz	w0, 19cc <main+0x57c>
		fclose(fo);
    1944:	aa1603e0 	mov	x0, x22
    1948:	97fffe3a 	bl	1230 <fclose@plt>
	}

	sptFreeTimer(timer);
    194c:	f94043e0 	ldr	x0, [sp, #128]
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1950:	d2800016 	mov	x22, #0x0                   	// #0
	sptFreeTimer(timer);
    1954:	94000ba7 	bl	47f0 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
		sptFreeMatrix(U[m]);
    1958:	f8767a60 	ldr	x0, [x19, x22, lsl #3]
	for(sptIndex m=0; m<nmodes; ++m) {
    195c:	910006d6 	add	x22, x22, #0x1
		sptFreeMatrix(U[m]);
    1960:	94000684 	bl	3370 <sptFreeMatrix>
	for(sptIndex m=0; m<nmodes; ++m) {
    1964:	6b16029f 	cmp	w20, w22
    1968:	54ffff88 	b.hi	1958 <main+0x508>  // b.pmore
	}
	sptFreeSparseTensor(&X);
    196c:	aa1903e0 	mov	x0, x25
    1970:	94000190 	bl	1fb0 <sptFreeSparseTensor>
	free(mats_order);
    1974:	aa1803e0 	mov	x0, x24
    1978:	97fffe8a 	bl	13a0 <free@plt>
	sptFreeMatrix(U[nmodes]);
    197c:	f87b7a60 	ldr	x0, [x19, x27, lsl #3]
    1980:	9400067c 	bl	3370 <sptFreeMatrix>
	free(U);
    1984:	aa1303e0 	mov	x0, x19
    1988:	97fffe86 	bl	13a0 <free@plt>

	if (!random){
    198c:	34000355 	cbz	w21, 19f4 <main+0x5a4>
		if(fPtr1 != NULL) fclose(fPtr1);
		if(fPtr2 != NULL) fclose(fPtr2);

	}
	return 0;
}
    1990:	52800000 	mov	w0, #0x0                   	// #0
    1994:	a9407bfd 	ldp	x29, x30, [sp]
    1998:	a94153f3 	ldp	x19, x20, [sp, #16]
    199c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    19a0:	a94363f7 	ldp	x23, x24, [sp, #48]
    19a4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    19a8:	a94573fb 	ldp	x27, x28, [sp, #80]
    19ac:	913203ff 	add	sp, sp, #0xc80
    19b0:	d65f03c0 	ret
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    19b4:	b94073e3 	ldr	w3, [sp, #112]
    19b8:	aa1803e2 	mov	x2, x24
    19bc:	aa1303e1 	mov	x1, x19
    19c0:	aa1903e0 	mov	x0, x25
    19c4:	940007b3 	bl	3890 <sptMTTKRP>
    19c8:	34fff120 	cbz	w0, 17ec <main+0x39c>
				sptAssert(fo != NULL);
    19cc:	12800000 	mov	w0, #0xffffffff            	// #-1
    19d0:	97fffdfc 	bl	11c0 <exit@plt>
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    19d4:	b94073e3 	ldr	w3, [sp, #112]
    19d8:	2a1c03e4 	mov	w4, w28
    19dc:	aa1803e2 	mov	x2, x24
    19e0:	aa1303e1 	mov	x1, x19
    19e4:	aa1903e0 	mov	x0, x25
    19e8:	94000a3b 	bl	42d4 <sptOmpMTTKRP>
    19ec:	34fff000 	cbz	w0, 17ec <main+0x39c>
    19f0:	17fffff7 	b	19cc <main+0x57c>
		FILE* fPtr1 = fopen(fvname, "r");
    19f4:	9112c3f6 	add	x22, sp, #0x4b0
    19f8:	90000033 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    19fc:	91006273 	add	x19, x19, #0x18
    1a00:	aa1603e0 	mov	x0, x22
    1a04:	aa1303e1 	mov	x1, x19
		FILE* fPtr2 = fopen(foname, "r");
    1a08:	912263f5 	add	x21, sp, #0x898
		FILE* fPtr1 = fopen(fvname, "r");
    1a0c:	97fffe0d 	bl	1240 <fopen@plt>
    1a10:	aa0003f4 	mov	x20, x0
		FILE* fPtr2 = fopen(foname, "r");
    1a14:	aa1303e1 	mov	x1, x19
    1a18:	aa1503e0 	mov	x0, x21
    1a1c:	97fffe09 	bl	1240 <fopen@plt>
    1a20:	aa0003f3 	mov	x19, x0
		if (fPtr1 == NULL || fPtr2 == NULL) {
    1a24:	f100029f 	cmp	x20, #0x0
    1a28:	fa401804 	ccmp	x0, #0x0, #0x4, ne  // ne = any
    1a2c:	54000200 	b.eq	1a6c <main+0x61c>  // b.none
			int diff = compareFile(fPtr1, fPtr2);
    1a30:	aa0003e1 	mov	x1, x0
    1a34:	aa1403e0 	mov	x0, x20
    1a38:	940000c2 	bl	1d40 <compareFile>
			if (diff == 0) {
    1a3c:	35000820 	cbnz	w0, 1b40 <main+0x6f0>
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
    1a40:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1a44:	aa1603e2 	mov	x2, x22
    1a48:	aa1503e1 	mov	x1, x21
    1a4c:	9101e000 	add	x0, x0, #0x78
    1a50:	97fffe6c 	bl	1400 <printf@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1a54:	aa1403e0 	mov	x0, x20
    1a58:	97fffdf6 	bl	1230 <fclose@plt>
		if(fPtr2 != NULL) fclose(fPtr2);
    1a5c:	b4fff9b3 	cbz	x19, 1990 <main+0x540>
    1a60:	aa1303e0 	mov	x0, x19
    1a64:	97fffdf3 	bl	1230 <fclose@plt>
	return 0;
    1a68:	17ffffca 	b	1990 <main+0x540>
			printf("\nUnable to open file.\n");
    1a6c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1a70:	91008000 	add	x0, x0, #0x20
    1a74:	97fffe3b 	bl	1360 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    1a78:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1a7c:	9100e000 	add	x0, x0, #0x38
    1a80:	97fffe38 	bl	1360 <puts@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    1a84:	b4fffed4 	cbz	x20, 1a5c <main+0x60c>
    1a88:	17fffff3 	b	1a54 <main+0x604>
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1a8c:	aa1803e2 	mov	x2, x24
    1a90:	aa1303e1 	mov	x1, x19
    1a94:	aa1903e0 	mov	x0, x25
    1a98:	9400077e 	bl	3890 <sptMTTKRP>
    1a9c:	34ffe860 	cbz	w0, 17a8 <main+0x358>
    1aa0:	17ffffcb 	b	19cc <main+0x57c>
		#pragma omp parallel
    1aa4:	52800024 	mov	w4, #0x1                   	// #1
    1aa8:	910203f7 	add	x23, sp, #0x80
    1aac:	52800003 	mov	w3, #0x0                   	// #0
    1ab0:	52800002 	mov	w2, #0x0                   	// #0
    1ab4:	aa1703e1 	mov	x1, x23
    1ab8:	90000000 	adrp	x0, 1000 <g_last_error+0x1000>
    1abc:	91328000 	add	x0, x0, #0xca0
    1ac0:	b90083e4 	str	w4, [sp, #128]
    1ac4:	97fffe57 	bl	1420 <GOMP_parallel@plt>
    1ac8:	b94083fc 	ldr	w28, [sp, #128]
        printf("\nnthreads: %d\n", nthreads);
    1acc:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1ad0:	913fc000 	add	x0, x0, #0xff0
    1ad4:	2a1c03e1 	mov	w1, w28
    1ad8:	97fffe4a 	bl	1400 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1adc:	b94073e3 	ldr	w3, [sp, #112]
    1ae0:	2a1c03e4 	mov	w4, w28
    1ae4:	aa1803e2 	mov	x2, x24
    1ae8:	aa1303e1 	mov	x1, x19
    1aec:	aa1903e0 	mov	x0, x25
    1af0:	940009f9 	bl	42d4 <sptOmpMTTKRP>
    1af4:	34ffe5e0 	cbz	w0, 17b0 <main+0x360>
    1af8:	17ffffb5 	b	19cc <main+0x57c>
	double gbw = (double)bytes / aver_time / 1e9;
    1afc:	9e630002 	ucvtf	d2, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1b00:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1b04:	91034000 	add	x0, x0, #0xd0
	double gbw = (double)bytes / aver_time / 1e9;
    1b08:	1e661842 	fdiv	d2, d2, d6
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1b0c:	1e611841 	fdiv	d1, d2, d1
    1b10:	97fffe3c 	bl	1400 <printf@plt>
	if(fo != NULL) {
    1b14:	b4000116 	cbz	x22, 1b34 <main+0x6e4>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1b18:	f94037e0 	ldr	x0, [sp, #104]
    1b1c:	aa1603e1 	mov	x1, x22
    1b20:	f8606a60 	ldr	x0, [x19, x0]
    1b24:	94000b4f 	bl	4860 <sptDumpMatrix>
    1b28:	35fff520 	cbnz	w0, 19cc <main+0x57c>
		fclose(fo);
    1b2c:	aa1603e0 	mov	x0, x22
    1b30:	97fffdc0 	bl	1230 <fclose@plt>
	sptFreeTimer(timer);
    1b34:	f94043e0 	ldr	x0, [sp, #128]
    1b38:	94000b2e 	bl	47f0 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1b3c:	17ffff8c 	b	196c <main+0x51c>
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1b40:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1b44:	91028000 	add	x0, x0, #0xa0
    1b48:	97fffe06 	bl	1360 <puts@plt>
    1b4c:	17ffffc2 	b	1a54 <main+0x604>
				print_usage(argv);
    1b50:	f9400280 	ldr	x0, [x20]
    1b54:	9400005c 	bl	1cc4 <print_usage.isra.0>
				exit(1);
    1b58:	52800020 	mov	w0, #0x1                   	// #1
    1b5c:	97fffd99 	bl	11c0 <exit@plt>

0000000000001b60 <init_have_lse_atomics>:
    1b60:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1b64:	d2800200 	mov	x0, #0x10                  	// #16
    1b68:	910003fd 	mov	x29, sp
    1b6c:	97fffdf5 	bl	1340 <__getauxval@plt>
    1b70:	53082000 	ubfx	w0, w0, #8, #1
    1b74:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1b78:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1b7c:	390aa420 	strb	w0, [x1, #681]
    1b80:	d65f03c0 	ret

0000000000001b84 <_start>:
    1b84:	d280001d 	mov	x29, #0x0                   	// #0
    1b88:	d280001e 	mov	x30, #0x0                   	// #0
    1b8c:	aa0003e5 	mov	x5, x0
    1b90:	f94003e1 	ldr	x1, [sp]
    1b94:	910023e2 	add	x2, sp, #0x8
    1b98:	910003e6 	mov	x6, sp
    1b9c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    1ba0:	f947ec00 	ldr	x0, [x0, #4056]
    1ba4:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0xfb88>
    1ba8:	f947e863 	ldr	x3, [x3, #4048]
    1bac:	b00000a4 	adrp	x4, 16000 <__FRAME_END__+0xfb88>
    1bb0:	f947cc84 	ldr	x4, [x4, #3992]
    1bb4:	97fffdb3 	bl	1280 <__libc_start_main@plt>
    1bb8:	97fffde6 	bl	1350 <abort@plt>

0000000000001bbc <call_weak_fn>:
    1bbc:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    1bc0:	f947e400 	ldr	x0, [x0, #4040]
    1bc4:	b4000040 	cbz	x0, 1bcc <call_weak_fn+0x10>
    1bc8:	17fffdda 	b	1330 <__gmon_start__@plt>
    1bcc:	d65f03c0 	ret

0000000000001bd0 <deregister_tm_clones>:
    1bd0:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1bd4:	910aa000 	add	x0, x0, #0x2a8
    1bd8:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1bdc:	910aa021 	add	x1, x1, #0x2a8
    1be0:	eb00003f 	cmp	x1, x0
    1be4:	540000c0 	b.eq	1bfc <deregister_tm_clones+0x2c>  // b.none
    1be8:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0xfb88>
    1bec:	f947d021 	ldr	x1, [x1, #4000]
    1bf0:	b4000061 	cbz	x1, 1bfc <deregister_tm_clones+0x2c>
    1bf4:	aa0103f0 	mov	x16, x1
    1bf8:	d61f0200 	br	x16
    1bfc:	d65f03c0 	ret

0000000000001c00 <register_tm_clones>:
    1c00:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1c04:	910aa000 	add	x0, x0, #0x2a8
    1c08:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1c0c:	910aa021 	add	x1, x1, #0x2a8
    1c10:	cb000021 	sub	x1, x1, x0
    1c14:	d37ffc22 	lsr	x2, x1, #63
    1c18:	8b810c41 	add	x1, x2, x1, asr #3
    1c1c:	9341fc21 	asr	x1, x1, #1
    1c20:	b40000c1 	cbz	x1, 1c38 <register_tm_clones+0x38>
    1c24:	b00000a2 	adrp	x2, 16000 <__FRAME_END__+0xfb88>
    1c28:	f947f042 	ldr	x2, [x2, #4064]
    1c2c:	b4000062 	cbz	x2, 1c38 <register_tm_clones+0x38>
    1c30:	aa0203f0 	mov	x16, x2
    1c34:	d61f0200 	br	x16
    1c38:	d65f03c0 	ret
    1c3c:	d503201f 	nop

0000000000001c40 <__do_global_dtors_aux>:
    1c40:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1c44:	910003fd 	mov	x29, sp
    1c48:	f9000bf3 	str	x19, [sp, #16]
    1c4c:	d00000b3 	adrp	x19, 17000 <memcpy@GLIBC_2.17>
    1c50:	394aa260 	ldrb	w0, [x19, #680]
    1c54:	35000140 	cbnz	w0, 1c7c <__do_global_dtors_aux+0x3c>
    1c58:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    1c5c:	f947d400 	ldr	x0, [x0, #4008]
    1c60:	b4000080 	cbz	x0, 1c70 <__do_global_dtors_aux+0x30>
    1c64:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1c68:	f940ac00 	ldr	x0, [x0, #344]
    1c6c:	97fffd59 	bl	11d0 <__cxa_finalize@plt>
    1c70:	97ffffd8 	bl	1bd0 <deregister_tm_clones>
    1c74:	52800020 	mov	w0, #0x1                   	// #1
    1c78:	390aa260 	strb	w0, [x19, #680]
    1c7c:	f9400bf3 	ldr	x19, [sp, #16]
    1c80:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1c84:	d65f03c0 	ret
    1c88:	d503201f 	nop
    1c8c:	d503201f 	nop

0000000000001c90 <frame_dummy>:
    1c90:	17ffffdc 	b	1c00 <register_tm_clones>
    1c94:	d503201f 	nop
    1c98:	d503201f 	nop
    1c9c:	d503201f 	nop

0000000000001ca0 <main._omp_fn.0>:
		#pragma omp parallel
    1ca0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1ca4:	910003fd 	mov	x29, sp
    1ca8:	f9000bf3 	str	x19, [sp, #16]
    1cac:	aa0003f3 	mov	x19, x0
            nthreads = omp_get_num_threads();
    1cb0:	97fffd94 	bl	1300 <omp_get_num_threads@plt>
    1cb4:	b9000260 	str	w0, [x19]
		#pragma omp parallel
    1cb8:	f9400bf3 	ldr	x19, [sp, #16]
    1cbc:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1cc0:	d65f03c0 	ret

0000000000001cc4 <print_usage.isra.0>:
static void print_usage(char ** argv) {
    1cc4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
	printf("Usage: %s [options] \n\n", argv[0]);
    1cc8:	aa0003e1 	mov	x1, x0
    1ccc:	f0000002 	adrp	x2, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
static void print_usage(char ** argv) {
    1cd0:	910003fd 	mov	x29, sp
	printf("Usage: %s [options] \n\n", argv[0]);
    1cd4:	91346040 	add	x0, x2, #0xd18
    1cd8:	97fffdca 	bl	1400 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1cdc:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1ce0:	9134c000 	add	x0, x0, #0xd30
    1ce4:	97fffd9f 	bl	1360 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    1ce8:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1cec:	91358000 	add	x0, x0, #0xd60
    1cf0:	97fffd9c 	bl	1360 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1cf4:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1cf8:	91366000 	add	x0, x0, #0xd98
    1cfc:	97fffd99 	bl	1360 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    1d00:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1d04:	91380000 	add	x0, x0, #0xe00
    1d08:	97fffd96 	bl	1360 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    1d0c:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1d10:	91394000 	add	x0, x0, #0xe50
    1d14:	97fffd93 	bl	1360 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    1d18:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1d1c:	913a4000 	add	x0, x0, #0xe90
    1d20:	97fffd90 	bl	1360 <puts@plt>
	printf("         --help\n");
    1d24:	f0000000 	adrp	x0, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    1d28:	913c8000 	add	x0, x0, #0xf20
    1d2c:	97fffd8d 	bl	1360 <puts@plt>
}
    1d30:	a8c17bfd 	ldp	x29, x30, [sp], #16
	printf("\n");
    1d34:	52800140 	mov	w0, #0xa                   	// #10
    1d38:	17fffdbe 	b	1430 <putchar@plt>
    1d3c:	d503201f 	nop

0000000000001d40 <compareFile>:



int compareFile(FILE * fPtr1, FILE * fPtr2)
{
    1d40:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1d44:	910003fd 	mov	x29, sp
    1d48:	a90153f3 	stp	x19, x20, [sp, #16]
    1d4c:	aa0103f4 	mov	x20, x1
    1d50:	f90013f5 	str	x21, [sp, #32]
    1d54:	aa0003f5 	mov	x21, x0
    1d58:	14000006 	b	1d70 <compareFile+0x30>
		ch1 = fgetc(fPtr1);
		ch2 = fgetc(fPtr2);
		if (ch1 != ch2) {
			return -1;
		}
	} while (ch1 != EOF && ch2 != EOF);
    1d5c:	3100067f 	cmn	w19, #0x1
    1d60:	1a9f07e2 	cset	w2, ne  // ne = any
    1d64:	7100005f 	cmp	w2, #0x0
    1d68:	3a411a64 	ccmn	w19, #0x1, #0x4, ne  // ne = any
    1d6c:	540001a0 	b.eq	1da0 <compareFile+0x60>  // b.none
		ch1 = fgetc(fPtr1);
    1d70:	aa1503e0 	mov	x0, x21
    1d74:	97fffd47 	bl	1290 <fgetc@plt>
    1d78:	2a0003f3 	mov	w19, w0
		ch2 = fgetc(fPtr2);
    1d7c:	aa1403e0 	mov	x0, x20
    1d80:	97fffd44 	bl	1290 <fgetc@plt>
		if (ch1 != ch2) {
    1d84:	6b00027f 	cmp	w19, w0
    1d88:	54fffea0 	b.eq	1d5c <compareFile+0x1c>  // b.none
			return -1;
    1d8c:	12800000 	mov	w0, #0xffffffff            	// #-1
	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
		return 0;
	else
		return -1;
}
    1d90:	a94153f3 	ldp	x19, x20, [sp, #16]
    1d94:	f94013f5 	ldr	x21, [sp, #32]
    1d98:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1d9c:	d65f03c0 	ret
	if (ch1 == EOF && ch2 == EOF)
    1da0:	4b0203e0 	neg	w0, w2
}
    1da4:	a94153f3 	ldp	x19, x20, [sp, #16]
    1da8:	f94013f5 	ldr	x21, [sp, #32]
    1dac:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1db0:	d65f03c0 	ret
    1db4:	d503201f 	nop
    1db8:	d503201f 	nop
    1dbc:	d503201f 	nop

0000000000001dc0 <sptNewSparseTensor>:
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1dc0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    1dc4:	910003fd 	mov	x29, sp
    1dc8:	a9025bf5 	stp	x21, x22, [sp, #32]
    1dcc:	aa0003f6 	mov	x22, x0
	}
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
//	spt_CheckOSError(!tsr->ndims, "SpTns New");
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
	tsr->nnz = 0;
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1dd0:	52800315 	mov	w21, #0x18                  	// #24
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1dd4:	a90153f3 	stp	x19, x20, [sp, #16]
    1dd8:	2a0103f3 	mov	w19, w1
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1ddc:	9bb57c35 	umull	x21, w1, w21
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1de0:	f9001bf7 	str	x23, [sp, #48]
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1de4:	d37e7e77 	ubfiz	x23, x19, #2, #32
	tsr->nmodes = nmodes;
    1de8:	b90002c1 	str	w1, [x22]
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1dec:	aa0203f4 	mov	x20, x2
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1df0:	aa1703e0 	mov	x0, x23
    1df4:	97fffd1b 	bl	1260 <malloc@plt>
    1df8:	f90006c0 	str	x0, [x22, #8]
	for(i = 0; i < nmodes; ++i) {
    1dfc:	34000a33 	cbz	w19, 1f40 <sptNewSparseTensor+0x180>
    1e00:	51000661 	sub	w1, w19, #0x1
    1e04:	7100143f 	cmp	w1, #0x5
    1e08:	54000ce9 	b.ls	1fa4 <sptNewSparseTensor+0x1e4>  // b.plast
    1e0c:	90000021 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    1e10:	53027e64 	lsr	w4, w19, #2
    1e14:	4f000482 	movi	v2.4s, #0x4
    1e18:	aa0003e3 	mov	x3, x0
    1e1c:	3dc06420 	ldr	q0, [x1, #400]
    1e20:	8b245004 	add	x4, x0, w4, uxtw #4
    1e24:	d503201f 	nop
    1e28:	4ea01c01 	mov	v1.16b, v0.16b
    1e2c:	4ea28400 	add	v0.4s, v0.4s, v2.4s
		tsr->sortorder[i] = i;
    1e30:	3c810461 	str	q1, [x3], #16
	for(i = 0; i < nmodes; ++i) {
    1e34:	eb04007f 	cmp	x3, x4
    1e38:	54ffff81 	b.ne	1e28 <sptNewSparseTensor+0x68>  // b.any
    1e3c:	121e7663 	and	w3, w19, #0xfffffffc
    1e40:	f240067f 	tst	x19, #0x3
    1e44:	54000320 	b.eq	1ea8 <sptNewSparseTensor+0xe8>  // b.none
		tsr->sortorder[i] = i;
    1e48:	2a0303e1 	mov	w1, w3
	for(i = 0; i < nmodes; ++i) {
    1e4c:	11000462 	add	w2, w3, #0x1
		tsr->sortorder[i] = i;
    1e50:	b8217803 	str	w3, [x0, x1, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1e54:	6b02027f 	cmp	w19, w2
    1e58:	54000289 	b.ls	1ea8 <sptNewSparseTensor+0xe8>  // b.plast
    1e5c:	11000861 	add	w1, w3, #0x2
		tsr->sortorder[i] = i;
    1e60:	b8225802 	str	w2, [x0, w2, uxtw #2]
	for(i = 0; i < nmodes; ++i) {
    1e64:	6b01027f 	cmp	w19, w1
    1e68:	54000209 	b.ls	1ea8 <sptNewSparseTensor+0xe8>  // b.plast
		tsr->sortorder[i] = i;
    1e6c:	2a0103e4 	mov	w4, w1
	for(i = 0; i < nmodes; ++i) {
    1e70:	11000c62 	add	w2, w3, #0x3
		tsr->sortorder[i] = i;
    1e74:	b8247801 	str	w1, [x0, x4, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1e78:	6b02027f 	cmp	w19, w2
    1e7c:	54000169 	b.ls	1ea8 <sptNewSparseTensor+0xe8>  // b.plast
    1e80:	11001061 	add	w1, w3, #0x4
		tsr->sortorder[i] = i;
    1e84:	b8225802 	str	w2, [x0, w2, uxtw #2]
	for(i = 0; i < nmodes; ++i) {
    1e88:	6b01027f 	cmp	w19, w1
    1e8c:	540000e9 	b.ls	1ea8 <sptNewSparseTensor+0xe8>  // b.plast
		tsr->sortorder[i] = i;
    1e90:	2a0103e2 	mov	w2, w1
	for(i = 0; i < nmodes; ++i) {
    1e94:	11001463 	add	w3, w3, #0x5
		tsr->sortorder[i] = i;
    1e98:	b8227801 	str	w1, [x0, x2, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1e9c:	6b03027f 	cmp	w19, w3
    1ea0:	54000049 	b.ls	1ea8 <sptNewSparseTensor+0xe8>  // b.plast
		tsr->sortorder[i] = i;
    1ea4:	b8235803 	str	w3, [x0, w3, uxtw #2]
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1ea8:	aa1703e0 	mov	x0, x23
    1eac:	97fffced 	bl	1260 <malloc@plt>
    1eb0:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1eb4:	aa1403e1 	mov	x1, x20
    1eb8:	aa1703e2 	mov	x2, x23
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1ebc:	d2800014 	mov	x20, #0x0                   	// #0
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1ec0:	97fffcbc 	bl	11b0 <memcpy@plt>
	tsr->nnz = 0;
    1ec4:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1ec8:	aa1503e0 	mov	x0, x21
    1ecc:	52800315 	mov	w21, #0x18                  	// #24
    1ed0:	97fffce4 	bl	1260 <malloc@plt>
    1ed4:	f90012c0 	str	x0, [x22, #32]
//	spt_CheckOSError(!tsr->inds, "SpTns New");
	for(i = 0; i < nmodes; ++i) {
    1ed8:	9bb57e75 	umull	x21, w19, w21
    1edc:	14000004 	b	1eec <sptNewSparseTensor+0x12c>
    1ee0:	eb15029f 	cmp	x20, x21
    1ee4:	54000420 	b.eq	1f68 <sptNewSparseTensor+0x1a8>  // b.none
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1ee8:	f94012c0 	ldr	x0, [x22, #32]
    1eec:	8b140000 	add	x0, x0, x20
    1ef0:	d2800002 	mov	x2, #0x0                   	// #0
    1ef4:	d2800001 	mov	x1, #0x0                   	// #0
	for(i = 0; i < nmodes; ++i) {
    1ef8:	91006294 	add	x20, x20, #0x18
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1efc:	940000fd 	bl	22f0 <sptNewIndexVector>
    1f00:	2a0003f3 	mov	w19, w0
		spt_CheckError(result, "SpTns New", NULL);
    1f04:	34fffee0 	cbz	w0, 1ee0 <sptNewSparseTensor+0x120>
    1f08:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    1f0c:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1f10:	91056042 	add	x2, x2, #0x158
    1f14:	91060000 	add	x0, x0, #0x180
    1f18:	2a1303e1 	mov	w1, w19
    1f1c:	d2800004 	mov	x4, #0x0                   	// #0
    1f20:	52800643 	mov	w3, #0x32                  	// #50
    1f24:	94000ab3 	bl	49f0 <spt_ComplainError>
	}
	result = sptNewValueVector(&tsr->values, 0, 0);
	spt_CheckError(result, "SpTns New", NULL);
	return 0;
}
    1f28:	2a1303e0 	mov	w0, w19
    1f2c:	a94153f3 	ldp	x19, x20, [sp, #16]
    1f30:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1f34:	f9401bf7 	ldr	x23, [sp, #48]
    1f38:	a8c47bfd 	ldp	x29, x30, [sp], #64
    1f3c:	d65f03c0 	ret
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1f40:	aa1703e0 	mov	x0, x23
    1f44:	97fffcc7 	bl	1260 <malloc@plt>
    1f48:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1f4c:	aa1703e2 	mov	x2, x23
    1f50:	aa1403e1 	mov	x1, x20
    1f54:	97fffc97 	bl	11b0 <memcpy@plt>
	tsr->nnz = 0;
    1f58:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1f5c:	aa1503e0 	mov	x0, x21
    1f60:	97fffcc0 	bl	1260 <malloc@plt>
    1f64:	f90012c0 	str	x0, [x22, #32]
	result = sptNewValueVector(&tsr->values, 0, 0);
    1f68:	9100a2c0 	add	x0, x22, #0x28
    1f6c:	d2800002 	mov	x2, #0x0                   	// #0
    1f70:	d2800001 	mov	x1, #0x0                   	// #0
    1f74:	9400002f 	bl	2030 <sptNewValueVector>
    1f78:	2a0003f3 	mov	w19, w0
	spt_CheckError(result, "SpTns New", NULL);
    1f7c:	34fffd60 	cbz	w0, 1f28 <sptNewSparseTensor+0x168>
    1f80:	2a1303e1 	mov	w1, w19
    1f84:	90000022 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    1f88:	90000020 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    1f8c:	91056042 	add	x2, x2, #0x158
    1f90:	91060000 	add	x0, x0, #0x180
    1f94:	d2800004 	mov	x4, #0x0                   	// #0
    1f98:	528006a3 	mov	w3, #0x35                  	// #53
    1f9c:	94000a95 	bl	49f0 <spt_ComplainError>
    1fa0:	17ffffe2 	b	1f28 <sptNewSparseTensor+0x168>
	for(i = 0; i < nmodes; ++i) {
    1fa4:	52800003 	mov	w3, #0x0                   	// #0
    1fa8:	17ffffa8 	b	1e48 <sptNewSparseTensor+0x88>
    1fac:	d503201f 	nop

0000000000001fb0 <sptFreeSparseTensor>:

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    1fb0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1fb4:	910003fd 	mov	x29, sp
    1fb8:	a90153f3 	stp	x19, x20, [sp, #16]
    1fbc:	aa0003f4 	mov	x20, x0
	sptIndex i;
	for(i = 0; i < tsr->nmodes; ++i) {
    1fc0:	b9400000 	ldr	w0, [x0]
    1fc4:	340001a0 	cbz	w0, 1ff8 <sptFreeSparseTensor+0x48>
    1fc8:	52800013 	mov	w19, #0x0                   	// #0
    1fcc:	f90013f5 	str	x21, [sp, #32]
		sptFreeIndexVector(&tsr->inds[i]);
    1fd0:	52800315 	mov	w21, #0x18                  	// #24
    1fd4:	d503201f 	nop
    1fd8:	f9401280 	ldr	x0, [x20, #32]
    1fdc:	9bb50260 	umaddl	x0, w19, w21, x0
	for(i = 0; i < tsr->nmodes; ++i) {
    1fe0:	11000673 	add	w19, w19, #0x1
		sptFreeIndexVector(&tsr->inds[i]);
    1fe4:	94000150 	bl	2524 <sptFreeIndexVector>
	for(i = 0; i < tsr->nmodes; ++i) {
    1fe8:	b9400280 	ldr	w0, [x20]
    1fec:	6b13001f 	cmp	w0, w19
    1ff0:	54ffff48 	b.hi	1fd8 <sptFreeSparseTensor+0x28>  // b.pmore
    1ff4:	f94013f5 	ldr	x21, [sp, #32]
	}
	free(tsr->sortorder);
    1ff8:	f9400680 	ldr	x0, [x20, #8]
    1ffc:	97fffce9 	bl	13a0 <free@plt>
	free(tsr->ndims);
    2000:	f9400a80 	ldr	x0, [x20, #16]
    2004:	97fffce7 	bl	13a0 <free@plt>
	free(tsr->inds);
    2008:	f9401280 	ldr	x0, [x20, #32]
    200c:	97fffce5 	bl	13a0 <free@plt>
	sptFreeValueVector(&tsr->values);
    2010:	9100a280 	add	x0, x20, #0x28
    2014:	940000b4 	bl	22e4 <sptFreeValueVector>
	tsr->nmodes = 0;
    2018:	b900029f 	str	wzr, [x20]
}
    201c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2020:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2024:	d65f03c0 	ret
    2028:	d503201f 	nop
    202c:	d503201f 	nop

0000000000002030 <sptNewValueVector>:
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    2030:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2034:	f100083f 	cmp	x1, #0x2
    2038:	d2800043 	mov	x3, #0x2                   	// #2
    203c:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    2040:	910003fd 	mov	x29, sp
    2044:	eb02007f 	cmp	x3, x2
    2048:	aa0103e4 	mov	x4, x1
    204c:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
    2050:	a90153f3 	stp	x19, x20, [sp, #16]
    2054:	aa0003f3 	mov	x19, x0
	if(cap < 2) {
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    2058:	d2800021 	mov	x1, #0x1                   	// #1
	vec->cap = cap;
    205c:	a9000804 	stp	x4, x2, [x0]
	vec->data = malloc(cap * sizeof *vec->data);
    2060:	d37ef440 	lsl	x0, x2, #2
    2064:	97fffc97 	bl	12c0 <calloc@plt>
    2068:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "ValVec New");
    206c:	b40000a0 	cbz	x0, 2080 <sptNewValueVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    2070:	52800000 	mov	w0, #0x0                   	// #0
}
    2074:	a94153f3 	ldp	x19, x20, [sp, #16]
    2078:	a8c27bfd 	ldp	x29, x30, [sp], #32
    207c:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "ValVec New");
    2080:	97fffce4 	bl	1410 <__errno_location@plt>
    2084:	aa0003f3 	mov	x19, x0
	return 0;
    2088:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "ValVec New");
    208c:	b9400274 	ldr	w20, [x19]
    2090:	3140429f 	cmn	w20, #0x10, lsl #12
    2094:	54ffff00 	b.eq	2074 <sptNewValueVector+0x44>  // b.none
    2098:	2a1403e0 	mov	w0, w20
    209c:	97fffc9d 	bl	1310 <strerror@plt>
    20a0:	11404281 	add	w1, w20, #0x10, lsl #12
    20a4:	aa0003e4 	mov	x4, x0
    20a8:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    20ac:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    20b0:	91068042 	add	x2, x2, #0x1a0
    20b4:	91072000 	add	x0, x0, #0x1c8
    20b8:	528005c3 	mov	w3, #0x2e                  	// #46
    20bc:	94000a4d 	bl	49f0 <spt_ComplainError>
    20c0:	b9400260 	ldr	w0, [x19]
    20c4:	11404000 	add	w0, w0, #0x10, lsl #12
    20c8:	17ffffeb 	b	2074 <sptNewValueVector+0x44>
    20cc:	d503201f 	nop

00000000000020d0 <sptConstantValueVector>:
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
	for(sptNnzIndex i=0; i<vec->len; ++i)
    20d0:	f9400002 	ldr	x2, [x0]
    20d4:	b4000302 	cbz	x2, 2134 <sptConstantValueVector+0x64>
    20d8:	d1000441 	sub	x1, x2, #0x1
		vec->data[i] = val;
    20dc:	f9400803 	ldr	x3, [x0, #16]
    20e0:	f100083f 	cmp	x1, #0x2
    20e4:	540002c9 	b.ls	213c <sptConstantValueVector+0x6c>  // b.plast
    20e8:	d342fc41 	lsr	x1, x2, #2
    20ec:	aa0303e0 	mov	x0, x3
    20f0:	4e040401 	dup	v1.4s, v0.s[0]
    20f4:	8b011061 	add	x1, x3, x1, lsl #4
    20f8:	3c810401 	str	q1, [x0], #16
	for(sptNnzIndex i=0; i<vec->len; ++i)
    20fc:	eb01001f 	cmp	x0, x1
    2100:	54ffffc1 	b.ne	20f8 <sptConstantValueVector+0x28>  // b.any
    2104:	927ef440 	and	x0, x2, #0xfffffffffffffffc
    2108:	f240045f 	tst	x2, #0x3
    210c:	54000140 	b.eq	2134 <sptConstantValueVector+0x64>  // b.none
    2110:	91000401 	add	x1, x0, #0x1
		vec->data[i] = val;
    2114:	bc207860 	str	s0, [x3, x0, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2118:	eb02003f 	cmp	x1, x2
    211c:	540000c2 	b.cs	2134 <sptConstantValueVector+0x64>  // b.hs, b.nlast
    2120:	91000800 	add	x0, x0, #0x2
		vec->data[i] = val;
    2124:	bc217860 	str	s0, [x3, x1, lsl #2]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    2128:	eb00005f 	cmp	x2, x0
    212c:	54000049 	b.ls	2134 <sptConstantValueVector+0x64>  // b.plast
		vec->data[i] = val;
    2130:	bc207860 	str	s0, [x3, x0, lsl #2]
	return 0;
}
    2134:	52800000 	mov	w0, #0x0                   	// #0
    2138:	d65f03c0 	ret
	for(sptNnzIndex i=0; i<vec->len; ++i)
    213c:	d2800000 	mov	x0, #0x0                   	// #0
    2140:	17fffff4 	b	2110 <sptConstantValueVector+0x40>

0000000000002144 <sptAppendValueVector>:
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    2144:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2148:	910003fd 	mov	x29, sp
    214c:	a90153f3 	stp	x19, x20, [sp, #16]
    2150:	aa0003f3 	mov	x19, x0
	if(vec->cap <= vec->len) {
    2154:	a9400001 	ldp	x1, x0, [x0]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2158:	f9400a74 	ldr	x20, [x19, #16]
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    215c:	fd001fe8 	str	d8, [sp, #56]
    2160:	1e204008 	fmov	s8, s0
	if(vec->cap <= vec->len) {
    2164:	eb01001f 	cmp	x0, x1
    2168:	54000129 	b.ls	218c <sptAppendValueVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "ValVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    216c:	bc217a88 	str	s8, [x20, x1, lsl #2]
	++vec->len;
	return 0;
    2170:	52800000 	mov	w0, #0x0                   	// #0
	++vec->len;
    2174:	91000421 	add	x1, x1, #0x1
    2178:	f9000261 	str	x1, [x19]
}
    217c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2180:	fd401fe8 	ldr	d8, [sp, #56]
    2184:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2188:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    218c:	a9025bf5 	stp	x21, x22, [sp, #32]
    2190:	8b400415 	add	x21, x0, x0, lsr #1
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2194:	aa1403e0 	mov	x0, x20
    2198:	d37ef6a1 	lsl	x1, x21, #2
    219c:	97fffc4d 	bl	12d0 <realloc@plt>
    21a0:	aa0003f4 	mov	x20, x0
		spt_CheckOSError(!newdata, "ValVec Append");
    21a4:	b40000a0 	cbz	x0, 21b8 <sptAppendValueVector+0x74>
		vec->data = newdata;
    21a8:	a900d275 	stp	x21, x20, [x19, #8]
	vec->data[vec->len] = value;
    21ac:	f9400261 	ldr	x1, [x19]
		vec->data = newdata;
    21b0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    21b4:	17ffffee 	b	216c <sptAppendValueVector+0x28>
		spt_CheckOSError(!newdata, "ValVec Append");
    21b8:	f9001bf7 	str	x23, [sp, #48]
    21bc:	97fffc95 	bl	1410 <__errno_location@plt>
    21c0:	b9400017 	ldr	w23, [x0]
    21c4:	aa0003f6 	mov	x22, x0
    21c8:	314042ff 	cmn	w23, #0x10, lsl #12
    21cc:	54000061 	b.ne	21d8 <sptAppendValueVector+0x94>  // b.any
    21d0:	f9401bf7 	ldr	x23, [sp, #48]
    21d4:	17fffff5 	b	21a8 <sptAppendValueVector+0x64>
    21d8:	2a1703e0 	mov	w0, w23
    21dc:	97fffc4d 	bl	1310 <strerror@plt>
    21e0:	114042e1 	add	w1, w23, #0x10, lsl #12
    21e4:	aa0003e4 	mov	x4, x0
    21e8:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    21ec:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    21f0:	91068042 	add	x2, x2, #0x1a0
    21f4:	91076000 	add	x0, x0, #0x1d8
    21f8:	52800a43 	mov	w3, #0x52                  	// #82
    21fc:	940009fd 	bl	49f0 <spt_ComplainError>
    2200:	b94002c0 	ldr	w0, [x22]
    2204:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2208:	11404000 	add	w0, w0, #0x10, lsl #12
    220c:	f9401bf7 	ldr	x23, [sp, #48]
    2210:	17ffffdb 	b	217c <sptAppendValueVector+0x38>

0000000000002214 <sptResizeValueVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2214:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2218:	f100083f 	cmp	x1, #0x2
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    221c:	910003fd 	mov	x29, sp
    2220:	a90153f3 	stp	x19, x20, [sp, #16]
    2224:	aa0003f3 	mov	x19, x0
    2228:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    222c:	f9400400 	ldr	x0, [x0, #8]
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2230:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2234:	d2800055 	mov	x21, #0x2                   	// #2
    2238:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    223c:	eb15001f 	cmp	x0, x21
    2240:	540000e1 	b.ne	225c <sptResizeValueVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    2244:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    2248:	f9000261 	str	x1, [x19]
}
    224c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2250:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2254:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2258:	d65f03c0 	ret
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    225c:	f9400a60 	ldr	x0, [x19, #16]
    2260:	d37ef6a1 	lsl	x1, x21, #2
    2264:	97fffc1b 	bl	12d0 <realloc@plt>
    2268:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "ValVec Resize");
    226c:	b4000100 	cbz	x0, 228c <sptResizeValueVector+0x78>
		vec->cap = newcap;
    2270:	a9005674 	stp	x20, x21, [x19]
	return 0;
    2274:	52800000 	mov	w0, #0x0                   	// #0
		vec->data = newdata;
    2278:	f9000a76 	str	x22, [x19, #16]
}
    227c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2280:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2284:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2288:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "ValVec Resize");
    228c:	a90363f7 	stp	x23, x24, [sp, #48]
    2290:	97fffc60 	bl	1410 <__errno_location@plt>
    2294:	b9400018 	ldr	w24, [x0]
    2298:	aa0003f7 	mov	x23, x0
    229c:	3140431f 	cmn	w24, #0x10, lsl #12
    22a0:	54000061 	b.ne	22ac <sptResizeValueVector+0x98>  // b.any
    22a4:	a94363f7 	ldp	x23, x24, [sp, #48]
    22a8:	17fffff2 	b	2270 <sptResizeValueVector+0x5c>
    22ac:	2a1803e0 	mov	w0, w24
    22b0:	97fffc18 	bl	1310 <strerror@plt>
    22b4:	11404301 	add	w1, w24, #0x10, lsl #12
    22b8:	aa0003e4 	mov	x4, x0
    22bc:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    22c0:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    22c4:	91068042 	add	x2, x2, #0x1a0
    22c8:	9107a000 	add	x0, x0, #0x1e8
    22cc:	52800d23 	mov	w3, #0x69                  	// #105
    22d0:	940009c8 	bl	49f0 <spt_ComplainError>
    22d4:	b94002e0 	ldr	w0, [x23]
    22d8:	a94363f7 	ldp	x23, x24, [sp, #48]
    22dc:	11404000 	add	w0, w0, #0x10, lsl #12
    22e0:	17ffffdb 	b	224c <sptResizeValueVector+0x38>

00000000000022e4 <sptFreeValueVector>:
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
	vec->len = 0;
    22e4:	a9007c1f 	stp	xzr, xzr, [x0]
	vec->cap = 0;
	free(vec->data);
    22e8:	f9400800 	ldr	x0, [x0, #16]
    22ec:	17fffc2d 	b	13a0 <free@plt>

00000000000022f0 <sptNewIndexVector>:
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    22f0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    22f4:	f100083f 	cmp	x1, #0x2
    22f8:	d2800043 	mov	x3, #0x2                   	// #2
    22fc:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    2300:	910003fd 	mov	x29, sp
    2304:	eb02007f 	cmp	x3, x2
    2308:	aa0103e4 	mov	x4, x1
    230c:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
    2310:	a90153f3 	stp	x19, x20, [sp, #16]
    2314:	aa0003f3 	mov	x19, x0
	if(cap < 2) {
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    2318:	d2800021 	mov	x1, #0x1                   	// #1
	vec->cap = cap;
    231c:	a9000804 	stp	x4, x2, [x0]
	vec->data = malloc(cap * sizeof *vec->data);
    2320:	d37ef440 	lsl	x0, x2, #2
    2324:	97fffbe7 	bl	12c0 <calloc@plt>
    2328:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "IdxVec New");
    232c:	b40000a0 	cbz	x0, 2340 <sptNewIndexVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    2330:	52800000 	mov	w0, #0x0                   	// #0
}
    2334:	a94153f3 	ldp	x19, x20, [sp, #16]
    2338:	a8c27bfd 	ldp	x29, x30, [sp], #32
    233c:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "IdxVec New");
    2340:	97fffc34 	bl	1410 <__errno_location@plt>
    2344:	aa0003f3 	mov	x19, x0
	return 0;
    2348:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "IdxVec New");
    234c:	b9400274 	ldr	w20, [x19]
    2350:	3140429f 	cmn	w20, #0x10, lsl #12
    2354:	54ffff00 	b.eq	2334 <sptNewIndexVector+0x44>  // b.none
    2358:	2a1403e0 	mov	w0, w20
    235c:	97fffbed 	bl	1310 <strerror@plt>
    2360:	11404281 	add	w1, w20, #0x10, lsl #12
    2364:	aa0003e4 	mov	x4, x0
    2368:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    236c:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2370:	91068042 	add	x2, x2, #0x1a0
    2374:	9107e000 	add	x0, x0, #0x1f8
    2378:	52801283 	mov	w3, #0x94                  	// #148
    237c:	9400099d 	bl	49f0 <spt_ComplainError>
    2380:	b9400260 	ldr	w0, [x19]
    2384:	11404000 	add	w0, w0, #0x10, lsl #12
    2388:	17ffffeb 	b	2334 <sptNewIndexVector+0x44>
    238c:	d503201f 	nop

0000000000002390 <sptAppendIndexVector>:
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    2390:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2394:	910003fd 	mov	x29, sp
    2398:	a90153f3 	stp	x19, x20, [sp, #16]
    239c:	aa0003f3 	mov	x19, x0
    23a0:	2a0103f4 	mov	w20, w1
	if(vec->cap <= vec->len) {
    23a4:	a9400002 	ldp	x2, x0, [x0]
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    23a8:	a9025bf5 	stp	x21, x22, [sp, #32]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    23ac:	f9400a75 	ldr	x21, [x19, #16]
	if(vec->cap <= vec->len) {
    23b0:	eb02001f 	cmp	x0, x2
    23b4:	54000129 	b.ls	23d8 <sptAppendIndexVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "IdxVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    23b8:	b8227ab4 	str	w20, [x21, x2, lsl #2]
	++vec->len;
	return 0;
    23bc:	52800000 	mov	w0, #0x0                   	// #0
	++vec->len;
    23c0:	91000442 	add	x2, x2, #0x1
    23c4:	f9000262 	str	x2, [x19]
}
    23c8:	a94153f3 	ldp	x19, x20, [sp, #16]
    23cc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    23d0:	a8c47bfd 	ldp	x29, x30, [sp], #64
    23d4:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    23d8:	8b400416 	add	x22, x0, x0, lsr #1
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    23dc:	aa1503e0 	mov	x0, x21
    23e0:	d37ef6c1 	lsl	x1, x22, #2
    23e4:	97fffbbb 	bl	12d0 <realloc@plt>
    23e8:	aa0003f5 	mov	x21, x0
		spt_CheckOSError(!newdata, "IdxVec Append");
    23ec:	b4000080 	cbz	x0, 23fc <sptAppendIndexVector+0x6c>
		vec->data = newdata;
    23f0:	a900d676 	stp	x22, x21, [x19, #8]
	vec->data[vec->len] = value;
    23f4:	f9400262 	ldr	x2, [x19]
    23f8:	17fffff0 	b	23b8 <sptAppendIndexVector+0x28>
		spt_CheckOSError(!newdata, "IdxVec Append");
    23fc:	a90363f7 	stp	x23, x24, [sp, #48]
    2400:	97fffc04 	bl	1410 <__errno_location@plt>
    2404:	b9400018 	ldr	w24, [x0]
    2408:	aa0003f7 	mov	x23, x0
    240c:	3140431f 	cmn	w24, #0x10, lsl #12
    2410:	54000061 	b.ne	241c <sptAppendIndexVector+0x8c>  // b.any
    2414:	a94363f7 	ldp	x23, x24, [sp, #48]
    2418:	17fffff6 	b	23f0 <sptAppendIndexVector+0x60>
    241c:	2a1803e0 	mov	w0, w24
    2420:	97fffbbc 	bl	1310 <strerror@plt>
    2424:	11404301 	add	w1, w24, #0x10, lsl #12
    2428:	aa0003e4 	mov	x4, x0
    242c:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    2430:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2434:	91068042 	add	x2, x2, #0x1a0
    2438:	91082000 	add	x0, x0, #0x208
    243c:	52801543 	mov	w3, #0xaa                  	// #170
    2440:	9400096c 	bl	49f0 <spt_ComplainError>
    2444:	b94002e0 	ldr	w0, [x23]
    2448:	a94363f7 	ldp	x23, x24, [sp, #48]
    244c:	11404000 	add	w0, w0, #0x10, lsl #12
    2450:	17ffffde 	b	23c8 <sptAppendIndexVector+0x38>

0000000000002454 <sptResizeIndexVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    2454:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2458:	f100083f 	cmp	x1, #0x2
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    245c:	910003fd 	mov	x29, sp
    2460:	a90153f3 	stp	x19, x20, [sp, #16]
    2464:	aa0003f3 	mov	x19, x0
    2468:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    246c:	f9400400 	ldr	x0, [x0, #8]
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    2470:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2474:	d2800055 	mov	x21, #0x2                   	// #2
    2478:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    247c:	eb15001f 	cmp	x0, x21
    2480:	540000e1 	b.ne	249c <sptResizeIndexVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    2484:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    2488:	f9000261 	str	x1, [x19]
}
    248c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2490:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2494:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2498:	d65f03c0 	ret
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    249c:	f9400a60 	ldr	x0, [x19, #16]
    24a0:	d37ef6a1 	lsl	x1, x21, #2
    24a4:	97fffb8b 	bl	12d0 <realloc@plt>
    24a8:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "IdxVec Resize");
    24ac:	b4000100 	cbz	x0, 24cc <sptResizeIndexVector+0x78>
		vec->cap = newcap;
    24b0:	a9005674 	stp	x20, x21, [x19]
	return 0;
    24b4:	52800000 	mov	w0, #0x0                   	// #0
		vec->data = newdata;
    24b8:	f9000a76 	str	x22, [x19, #16]
}
    24bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    24c0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    24c4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    24c8:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "IdxVec Resize");
    24cc:	a90363f7 	stp	x23, x24, [sp, #48]
    24d0:	97fffbd0 	bl	1410 <__errno_location@plt>
    24d4:	b9400018 	ldr	w24, [x0]
    24d8:	aa0003f7 	mov	x23, x0
    24dc:	3140431f 	cmn	w24, #0x10, lsl #12
    24e0:	54000061 	b.ne	24ec <sptResizeIndexVector+0x98>  // b.any
    24e4:	a94363f7 	ldp	x23, x24, [sp, #48]
    24e8:	17fffff2 	b	24b0 <sptResizeIndexVector+0x5c>
    24ec:	2a1803e0 	mov	w0, w24
    24f0:	97fffb88 	bl	1310 <strerror@plt>
    24f4:	11404301 	add	w1, w24, #0x10, lsl #12
    24f8:	aa0003e4 	mov	x4, x0
    24fc:	f0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    2500:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2504:	91068042 	add	x2, x2, #0x1a0
    2508:	91086000 	add	x0, x0, #0x218
    250c:	52801823 	mov	w3, #0xc1                  	// #193
    2510:	94000938 	bl	49f0 <spt_ComplainError>
    2514:	b94002e0 	ldr	w0, [x23]
    2518:	a94363f7 	ldp	x23, x24, [sp, #48]
    251c:	11404000 	add	w0, w0, #0x10, lsl #12
    2520:	17ffffdb 	b	248c <sptResizeIndexVector+0x38>

0000000000002524 <sptFreeIndexVector>:
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    2524:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2528:	910003fd 	mov	x29, sp
    252c:	f9000bf3 	str	x19, [sp, #16]
    2530:	aa0003f3 	mov	x19, x0
	free(vec->data);
    2534:	f9400800 	ldr	x0, [x0, #16]
    2538:	97fffb9a 	bl	13a0 <free@plt>
	vec->len = 0;
    253c:	a9007e7f 	stp	xzr, xzr, [x19]
	vec->cap = 0;
}
    2540:	f9400bf3 	ldr	x19, [sp, #16]
    2544:	a8c27bfd 	ldp	x29, x30, [sp], #32
    2548:	d65f03c0 	ret
    254c:	d503201f 	nop

0000000000002550 <fill_binary_nnzidx._omp_fn.0>:
		sptIndex const BUF_LEN = 1024*1024;
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    2550:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2554:	910003fd 	mov	x29, sp
    2558:	f90013f5 	str	x21, [sp, #32]
    255c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2560:	34000355 	cbz	w21, 25c8 <fill_binary_nnzidx._omp_fn.0+0x78>
    2564:	a90153f3 	stp	x19, x20, [sp, #16]
    2568:	aa0003f4 	mov	x20, x0
    256c:	97fffb65 	bl	1300 <omp_get_num_threads@plt>
    2570:	2a0003f3 	mov	w19, w0
    2574:	97fffb2b 	bl	1220 <omp_get_thread_num@plt>
    2578:	1ad30aa2 	udiv	w2, w21, w19
    257c:	1b13d441 	msub	w1, w2, w19, w21
    2580:	6b01001f 	cmp	w0, w1
    2584:	54000283 	b.cc	25d4 <fill_binary_nnzidx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2588:	1b000441 	madd	w1, w2, w0, w1
    258c:	0b010043 	add	w3, w2, w1
    2590:	6b03003f 	cmp	w1, w3
    2594:	54000182 	b.cs	25c4 <fill_binary_nnzidx._omp_fn.0+0x74>  // b.hs, b.nlast
    2598:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    259c:	b9401284 	ldr	w4, [x20, #16]
    25a0:	0b010080 	add	w0, w4, w1
    25a4:	0b030083 	add	w3, w4, w3
    25a8:	8b214842 	add	x2, x2, w1, uxtw #2
    25ac:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    25b0:	b8404441 	ldr	w1, [x2], #4
    25b4:	f82058a1 	str	x1, [x5, w0, uxtw #3]
    25b8:	11000400 	add	w0, w0, #0x1
    25bc:	6b00007f 	cmp	w3, w0
    25c0:	54ffff81 	b.ne	25b0 <fill_binary_nnzidx._omp_fn.0+0x60>  // b.any
    25c4:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    25c8:	f94013f5 	ldr	x21, [sp, #32]
    25cc:	a8c37bfd 	ldp	x29, x30, [sp], #48
    25d0:	d65f03c0 	ret
    25d4:	11000442 	add	w2, w2, #0x1
    25d8:	52800001 	mov	w1, #0x0                   	// #0
    25dc:	17ffffeb 	b	2588 <fill_binary_nnzidx._omp_fn.0+0x38>

00000000000025e0 <fill_binary_idx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    25e0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    25e4:	910003fd 	mov	x29, sp
    25e8:	f90013f5 	str	x21, [sp, #32]
    25ec:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    25f0:	34000355 	cbz	w21, 2658 <fill_binary_idx._omp_fn.0+0x78>
    25f4:	a90153f3 	stp	x19, x20, [sp, #16]
    25f8:	aa0003f4 	mov	x20, x0
    25fc:	97fffb41 	bl	1300 <omp_get_num_threads@plt>
    2600:	2a0003f3 	mov	w19, w0
    2604:	97fffb07 	bl	1220 <omp_get_thread_num@plt>
    2608:	1ad30aa2 	udiv	w2, w21, w19
    260c:	1b13d441 	msub	w1, w2, w19, w21
    2610:	6b01001f 	cmp	w0, w1
    2614:	54000283 	b.cc	2664 <fill_binary_idx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2618:	1b000441 	madd	w1, w2, w0, w1
    261c:	0b010043 	add	w3, w2, w1
    2620:	6b03003f 	cmp	w1, w3
    2624:	54000182 	b.cs	2654 <fill_binary_idx._omp_fn.0+0x74>  // b.hs, b.nlast
    2628:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    262c:	b9401284 	ldr	w4, [x20, #16]
    2630:	0b010080 	add	w0, w4, w1
    2634:	0b030083 	add	w3, w4, w3
    2638:	8b214842 	add	x2, x2, w1, uxtw #2
    263c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2640:	b8404441 	ldr	w1, [x2], #4
    2644:	b82058a1 	str	w1, [x5, w0, uxtw #2]
    2648:	11000400 	add	w0, w0, #0x1
    264c:	6b00007f 	cmp	w3, w0
    2650:	54ffff81 	b.ne	2640 <fill_binary_idx._omp_fn.0+0x60>  // b.any
    2654:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    2658:	f94013f5 	ldr	x21, [sp, #32]
    265c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2660:	d65f03c0 	ret
    2664:	11000442 	add	w2, w2, #0x1
    2668:	52800001 	mov	w1, #0x0                   	// #0
    266c:	17ffffeb 	b	2618 <fill_binary_idx._omp_fn.0+0x38>

0000000000002670 <fill_binary_val._omp_fn.0>:
#endif

		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    2670:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2674:	910003fd 	mov	x29, sp
    2678:	f90013f5 	str	x21, [sp, #32]
    267c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2680:	34000375 	cbz	w21, 26ec <fill_binary_val._omp_fn.0+0x7c>
    2684:	a90153f3 	stp	x19, x20, [sp, #16]
    2688:	aa0003f4 	mov	x20, x0
    268c:	97fffb1d 	bl	1300 <omp_get_num_threads@plt>
    2690:	2a0003f3 	mov	w19, w0
    2694:	97fffae3 	bl	1220 <omp_get_thread_num@plt>
    2698:	1ad30aa2 	udiv	w2, w21, w19
    269c:	1b13d441 	msub	w1, w2, w19, w21
    26a0:	6b01001f 	cmp	w0, w1
    26a4:	540002a3 	b.cc	26f8 <fill_binary_val._omp_fn.0+0x88>  // b.lo, b.ul, b.last
    26a8:	1b000441 	madd	w1, w2, w0, w1
    26ac:	0b010043 	add	w3, w2, w1
    26b0:	6b03003f 	cmp	w1, w3
    26b4:	540001a2 	b.cs	26e8 <fill_binary_val._omp_fn.0+0x78>  // b.hs, b.nlast
    26b8:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    26bc:	b9401284 	ldr	w4, [x20, #16]
    26c0:	0b010080 	add	w0, w4, w1
    26c4:	0b030083 	add	w3, w4, w3
    26c8:	8b214c42 	add	x2, x2, w1, uxtw #3
    26cc:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    26d0:	fc408440 	ldr	d0, [x2], #8
    26d4:	1e624000 	fcvt	s0, d0
    26d8:	bc2058a0 	str	s0, [x5, w0, uxtw #2]
    26dc:	11000400 	add	w0, w0, #0x1
    26e0:	6b00007f 	cmp	w3, w0
    26e4:	54ffff61 	b.ne	26d0 <fill_binary_val._omp_fn.0+0x60>  // b.any
    26e8:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    26ec:	f94013f5 	ldr	x21, [sp, #32]
    26f0:	a8c37bfd 	ldp	x29, x30, [sp], #48
    26f4:	d65f03c0 	ret
    26f8:	11000442 	add	w2, w2, #0x1
    26fc:	52800001 	mov	w1, #0x0                   	// #0
    2700:	17ffffea 	b	26a8 <fill_binary_val._omp_fn.0+0x38>

0000000000002704 <sptLoadSparseTensor>:
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname)
{
    2704:	a9b57bfd 	stp	x29, x30, [sp, #-176]!
    2708:	910003fd 	mov	x29, sp
    270c:	a90153f3 	stp	x19, x20, [sp, #16]
    2710:	aa0203f3 	mov	x19, x2
    2714:	aa0003f4 	mov	x20, x0
	FILE * fp = fopen(fname, "r");
    2718:	aa0203e0 	mov	x0, x2
{
    271c:	a90363f7 	stp	x23, x24, [sp, #48]
    2720:	2a0103f7 	mov	w23, w1
	FILE * fp = fopen(fname, "r");
    2724:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    2728:	91006021 	add	x1, x1, #0x18
    272c:	97fffac5 	bl	1240 <fopen@plt>
	sptAssert(fp != NULL);
    2730:	a9025bf5 	stp	x21, x22, [sp, #32]
    2734:	b4004ea0 	cbz	x0, 3108 <sptLoadSparseTensor+0xa04>
	char const * const suffix = strrchr(fname, '.');
    2738:	aa0003f5 	mov	x21, x0
    273c:	528005c1 	mov	w1, #0x2e                  	// #46
    2740:	aa1303e0 	mov	x0, x19
    2744:	97fffaf7 	bl	1320 <strrchr@plt>
    2748:	aa0003f6 	mov	x22, x0
	if(suffix == NULL) {
    274c:	b4000200 	cbz	x0, 278c <sptLoadSparseTensor+0x88>
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    2750:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    2754:	9108a021 	add	x1, x1, #0x228
    2758:	97fffb0a 	bl	1380 <strcmp@plt>
    275c:	34000a20 	cbz	w0, 28a0 <sptLoadSparseTensor+0x19c>
    2760:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    2764:	aa1603e0 	mov	x0, x22
    2768:	9108c021 	add	x1, x1, #0x230
    276c:	97fffb05 	bl	1380 <strcmp@plt>
    2770:	34004140 	cbz	w0, 2f98 <sptLoadSparseTensor+0x894>
    2774:	aa1603e0 	mov	x0, x22
    2778:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    277c:	9108e021 	add	x1, x1, #0x238
    2780:	97fffb00 	bl	1380 <strcmp@plt>
	} while(file_extensions[++idx].extension != NULL);
    2784:	d2800041 	mov	x1, #0x2                   	// #2
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    2788:	340008e0 	cbz	w0, 28a4 <sptLoadSparseTensor+0x1a0>
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    278c:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    2790:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    2794:	aa1303e2 	mov	x2, x19
    2798:	91090021 	add	x1, x1, #0x240
    279c:	f947d800 	ldr	x0, [x0, #4016]
    27a0:	f9400000 	ldr	x0, [x0]
    27a4:	97fffb27 	bl	1440 <fprintf@plt>
	iores = fscanf(fp, "%u", &tsr->nmodes);
    27a8:	d0000018 	adrp	x24, 4000 <sptOmpMTTKRP_3D._omp_fn.0+0xd0>
    27ac:	aa1403e2 	mov	x2, x20
    27b0:	913de301 	add	x1, x24, #0xf78
    27b4:	aa1503e0 	mov	x0, x21
    27b8:	97fffaae 	bl	1270 <__isoc99_fscanf@plt>
	spt_CheckOSError(iores < 0, "SpTns Load");
    27bc:	37f83ac0 	tbnz	w0, #31, 2f14 <sptLoadSparseTensor+0x810>
	tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    27c0:	b9400296 	ldr	w22, [x20]
    27c4:	d2800021 	mov	x1, #0x1                   	// #1
    27c8:	a9046bf9 	stp	x25, x26, [sp, #64]
    27cc:	d37e7ed3 	ubfiz	x19, x22, #2, #32
    27d0:	aa1303e0 	mov	x0, x19
    27d4:	97fffabb 	bl	12c0 <calloc@plt>
    27d8:	f9000680 	str	x0, [x20, #8]
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    27dc:	b4003f00 	cbz	x0, 2fbc <sptLoadSparseTensor+0x8b8>
	tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    27e0:	aa1303e0 	mov	x0, x19
    27e4:	97fffa9f 	bl	1260 <malloc@plt>
    27e8:	f9000a80 	str	x0, [x20, #16]
    27ec:	aa0003e2 	mov	x2, x0
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    27f0:	b40040e0 	cbz	x0, 300c <sptLoadSparseTensor+0x908>
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    27f4:	913de319 	add	x25, x24, #0xf78
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    27f8:	52800013 	mov	w19, #0x0                   	// #0
    27fc:	350000f6 	cbnz	w22, 2818 <sptLoadSparseTensor+0x114>
    2800:	14000172 	b	2dc8 <sptLoadSparseTensor+0x6c4>
    2804:	b9400296 	ldr	w22, [x20]
    2808:	11000673 	add	w19, w19, #0x1
    280c:	6b16027f 	cmp	w19, w22
    2810:	54001762 	b.cs	2afc <sptLoadSparseTensor+0x3f8>  // b.hs, b.nlast
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2814:	f9400a82 	ldr	x2, [x20, #16]
    2818:	8b334842 	add	x2, x2, w19, uxtw #2
    281c:	aa1903e1 	mov	x1, x25
    2820:	aa1503e0 	mov	x0, x21
    2824:	97fffa93 	bl	1270 <__isoc99_fscanf@plt>
		spt_CheckOSError(iores != 1, "SpTns Load");
    2828:	7100041f 	cmp	w0, #0x1
    282c:	54fffec0 	b.eq	2804 <sptLoadSparseTensor+0x100>  // b.none
    2830:	97fffaf8 	bl	1410 <__errno_location@plt>
    2834:	b940001a 	ldr	w26, [x0]
    2838:	aa0003f6 	mov	x22, x0
    283c:	3140435f 	cmn	w26, #0x10, lsl #12
    2840:	54fffe20 	b.eq	2804 <sptLoadSparseTensor+0x100>  // b.none
    2844:	2a1a03e0 	mov	w0, w26
    2848:	97fffab2 	bl	1310 <strerror@plt>
    284c:	11404341 	add	w1, w26, #0x10, lsl #12
    2850:	aa0003e4 	mov	x4, x0
    2854:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    2858:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    285c:	910a6262 	add	x2, x19, #0x298
    2860:	910b0280 	add	x0, x20, #0x2c0
    2864:	52800a63 	mov	w3, #0x53                  	// #83
    2868:	94000862 	bl	49f0 <spt_ComplainError>
    286c:	b94002c0 	ldr	w0, [x22]
    2870:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2874:	11404000 	add	w0, w0, #0x10, lsl #12

	int iores;
	switch(get_file_type(fname)) {
		case 0:
			iores = p_tt_read_file(tsr, start_index, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
    2878:	350017e0 	cbnz	w0, 2b74 <sptLoadSparseTensor+0x470>
    287c:	d503201f 	nop
			iores = p_tt_read_binary_file(tsr, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
			break;
	}

	fclose(fp);
    2880:	aa1503e0 	mov	x0, x21
    2884:	97fffa6b 	bl	1230 <fclose@plt>

	return 0;
    2888:	52800000 	mov	w0, #0x0                   	// #0
}
    288c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2890:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2894:	a94363f7 	ldp	x23, x24, [sp, #48]
    2898:	a8cb7bfd 	ldp	x29, x30, [sp], #176
    289c:	d65f03c0 	ret
	size_t idx = 0;
    28a0:	d2800001 	mov	x1, #0x0                   	// #0
			return file_extensions[idx].type;
    28a4:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    28a8:	91350000 	add	x0, x0, #0xd40
    28ac:	8b011000 	add	x0, x0, x1, lsl #4
    28b0:	b9400800 	ldr	w0, [x0, #8]
	switch(get_file_type(fname)) {
    28b4:	34fff7a0 	cbz	w0, 27a8 <sptLoadSparseTensor+0xa4>
    28b8:	7100041f 	cmp	w0, #0x1
    28bc:	54fffe21 	b.ne	2880 <sptLoadSparseTensor+0x17c>  // b.any
	fread(&(header->magic), sizeof(header->magic), 1, fin);
    28c0:	aa1503e3 	mov	x3, x21
    28c4:	d2800022 	mov	x2, #0x1                   	// #1
    28c8:	d2800081 	mov	x1, #0x4                   	// #4
    28cc:	910203e0 	add	x0, sp, #0x80
    28d0:	a9046bf9 	stp	x25, x26, [sp, #64]
    28d4:	a90573fb 	stp	x27, x28, [sp, #80]
    28d8:	97fffaae 	bl	1390 <fread@plt>
	fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
    28dc:	aa1503e3 	mov	x3, x21
    28e0:	d2800022 	mov	x2, #0x1                   	// #1
    28e4:	d2800101 	mov	x1, #0x8                   	// #8
    28e8:	910223e0 	add	x0, sp, #0x88
    28ec:	97fffaa9 	bl	1390 <fread@plt>
	fread(&(header->val_width), sizeof(header->val_width), 1, fin);
    28f0:	d2800101 	mov	x1, #0x8                   	// #8
    28f4:	aa1503e3 	mov	x3, x21
    28f8:	910243e0 	add	x0, sp, #0x90
    28fc:	d2800022 	mov	x2, #0x1                   	// #1
    2900:	97fffaa4 	bl	1390 <fread@plt>
	if(header->idx_width > PASTA_INDEX_TYPEWIDTH / 8) {
    2904:	f94047e1 	ldr	x1, [sp, #136]
    2908:	f100103f 	cmp	x1, #0x4
    290c:	54004068 	b.hi	3118 <sptLoadSparseTensor+0xa14>  // b.pmore
	if(header->val_width > PASTA_VALUE_TYPEWIDTH / 8) {
    2910:	f9404be0 	ldr	x0, [sp, #144]
    2914:	f100101f 	cmp	x0, #0x4
    2918:	54002448 	b.hi	2da0 <sptLoadSparseTensor+0x69c>  // b.pmore
	sptIndex nmodes = 0;
    291c:	b90077ff 	str	wzr, [sp, #116]
	sptNnzIndex nnz = 0;
    2920:	f9003fff 	str	xzr, [sp, #120]
	if(header->idx_width == sizeof(sptIndex)) {
    2924:	f100103f 	cmp	x1, #0x4
    2928:	54002180 	b.eq	2d58 <sptLoadSparseTensor+0x654>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    292c:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2930:	97fffa4c 	bl	1260 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2934:	aa1503e3 	mov	x3, x21
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2938:	aa0003f3 	mov	x19, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    293c:	d2800022 	mov	x2, #0x1                   	// #1
    2940:	d2800081 	mov	x1, #0x4                   	// #4
    2944:	97fffa93 	bl	1390 <fread@plt>
    2948:	9000001b 	adrp	x27, 2000 <sptFreeSparseTensor+0x50>
#pragma omp parallel for schedule(static)
    294c:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2950:	9101d3e4 	add	x4, sp, #0x74
    2954:	910263e1 	add	x1, sp, #0x98
    2958:	52800003 	mov	w3, #0x0                   	// #0
    295c:	fd41e400 	ldr	d0, [x0, #968]
    2960:	52800002 	mov	w2, #0x0                   	// #0
    2964:	91178360 	add	x0, x27, #0x5e0
    2968:	a909cfe4 	stp	x4, x19, [sp, #152]
    296c:	fd0057e0 	str	d0, [sp, #168]
    2970:	97fffaac 	bl	1420 <GOMP_parallel@plt>
		free(ubuf);
    2974:	aa1303e0 	mov	x0, x19
    2978:	97fffa8a 	bl	13a0 <free@plt>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    297c:	b94077f7 	ldr	w23, [sp, #116]
    2980:	2a1703f6 	mov	w22, w23
    2984:	d37e7ee0 	ubfiz	x0, x23, #2, #32
    2988:	97fffa36 	bl	1260 <malloc@plt>
    298c:	f94047f3 	ldr	x19, [sp, #136]
    2990:	aa0003fa 	mov	x26, x0
	if(header->idx_width == sizeof(sptIndex)) {
    2994:	f100127f 	cmp	x19, #0x4
    2998:	54001f80 	b.eq	2d88 <sptLoadSparseTensor+0x684>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    299c:	d2a00800 	mov	x0, #0x400000              	// #4194304
    29a0:	97fffa30 	bl	1260 <malloc@plt>
    29a4:	aa0003f8 	mov	x24, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    29a8:	34000337 	cbz	w23, 2a0c <sptLoadSparseTensor+0x308>
    29ac:	9000001b 	adrp	x27, 2000 <sptFreeSparseTensor+0x50>
    29b0:	910263f9 	add	x25, sp, #0x98
    29b4:	9117837b 	add	x27, x27, #0x5e0
    29b8:	52800016 	mov	w22, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    29bc:	52a0021c 	mov	w28, #0x100000              	// #1048576
    29c0:	4b1602f3 	sub	w19, w23, w22
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    29c4:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    29c8:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    29cc:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    29d0:	1a9c9273 	csel	w19, w19, w28, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    29d4:	aa1803e0 	mov	x0, x24
    29d8:	2a1303e2 	mov	w2, w19
    29dc:	97fffa6d 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    29e0:	a909e3fa 	stp	x26, x24, [sp, #152]
    29e4:	aa1903e1 	mov	x1, x25
    29e8:	aa1b03e0 	mov	x0, x27
    29ec:	52800003 	mov	w3, #0x0                   	// #0
    29f0:	52800002 	mov	w2, #0x0                   	// #0
    29f4:	29154ff6 	stp	w22, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    29f8:	114402d6 	add	w22, w22, #0x100, lsl #12
    29fc:	97fffa89 	bl	1420 <GOMP_parallel@plt>
    2a00:	6b1602ff 	cmp	w23, w22
    2a04:	54fffde8 	b.hi	29c0 <sptLoadSparseTensor+0x2bc>  // b.pmore
    2a08:	f94047f3 	ldr	x19, [sp, #136]
		free(ubuf);
    2a0c:	aa1803e0 	mov	x0, x24
    2a10:	97fffa64 	bl	13a0 <free@plt>
	if(header->idx_width == sizeof(sptNnzIndex)) {
    2a14:	f100227f 	cmp	x19, #0x8
    2a18:	54001940 	b.eq	2d40 <sptLoadSparseTensor+0x63c>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2a1c:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2a20:	97fffa10 	bl	1260 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a24:	aa1503e3 	mov	x3, x21
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2a28:	aa0003f3 	mov	x19, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2a2c:	d2800022 	mov	x2, #0x1                   	// #1
    2a30:	d2800081 	mov	x1, #0x4                   	// #4
    2a34:	97fffa57 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    2a38:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2a3c:	9101e3e4 	add	x4, sp, #0x78
    2a40:	910263e1 	add	x1, sp, #0x98
    2a44:	52800003 	mov	w3, #0x0                   	// #0
    2a48:	fd41e400 	ldr	d0, [x0, #968]
    2a4c:	52800002 	mov	w2, #0x0                   	// #0
    2a50:	90000000 	adrp	x0, 2000 <sptFreeSparseTensor+0x50>
    2a54:	91154000 	add	x0, x0, #0x550
    2a58:	a909cfe4 	stp	x4, x19, [sp, #152]
    2a5c:	fd0057e0 	str	d0, [sp, #168]
    2a60:	97fffa70 	bl	1420 <GOMP_parallel@plt>
		free(ubuf);
    2a64:	aa1303e0 	mov	x0, x19
    2a68:	97fffa4e 	bl	13a0 <free@plt>
	sptNewSparseTensor(tsr, nmodes, dims);
    2a6c:	b94077e1 	ldr	w1, [sp, #116]
    2a70:	aa1a03e2 	mov	x2, x26
    2a74:	aa1403e0 	mov	x0, x20
    2a78:	97fffcd2 	bl	1dc0 <sptNewSparseTensor>
	for(sptIndex m=0; m < nmodes; ++m) {
    2a7c:	b94077e0 	ldr	w0, [sp, #116]
	tsr->nnz = nnz;
    2a80:	f9403fe1 	ldr	x1, [sp, #120]
    2a84:	f9000e81 	str	x1, [x20, #24]
	for(sptIndex m=0; m < nmodes; ++m) {
    2a88:	34000960 	cbz	w0, 2bb4 <sptLoadSparseTensor+0x4b0>
    2a8c:	52800013 	mov	w19, #0x0                   	// #0
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2a90:	52800316 	mov	w22, #0x18                  	// #24
    2a94:	14000006 	b	2aac <sptLoadSparseTensor+0x3a8>
	for(sptIndex m=0; m < nmodes; ++m) {
    2a98:	b94077e0 	ldr	w0, [sp, #116]
    2a9c:	11000673 	add	w19, w19, #0x1
	result = sptResizeValueVector(&tsr->values, nnz);
    2aa0:	f9403fe1 	ldr	x1, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2aa4:	6b00027f 	cmp	w19, w0
    2aa8:	54000862 	b.cs	2bb4 <sptLoadSparseTensor+0x4b0>  // b.hs, b.nlast
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2aac:	f9401280 	ldr	x0, [x20, #32]
    2ab0:	9bb60260 	umaddl	x0, w19, w22, x0
    2ab4:	97fffe68 	bl	2454 <sptResizeIndexVector>
		spt_CheckError(result, "SpTns Read", NULL);
    2ab8:	34ffff00 	cbz	w0, 2a98 <sptLoadSparseTensor+0x394>
    2abc:	2a0003e1 	mov	w1, w0
    2ac0:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    2ac4:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2ac8:	910a6262 	add	x2, x19, #0x298
    2acc:	910ee000 	add	x0, x0, #0x3b8
    2ad0:	d2800004 	mov	x4, #0x0                   	// #0
    2ad4:	528020a3 	mov	w3, #0x105                 	// #261
    2ad8:	940007c6 	bl	49f0 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2adc:	97fffa4d 	bl	1410 <__errno_location@plt>
    2ae0:	b9400016 	ldr	w22, [x0]
    2ae4:	aa0003f4 	mov	x20, x0
    2ae8:	314042df 	cmn	w22, #0x10, lsl #12
    2aec:	540023a1 	b.ne	2f60 <sptLoadSparseTensor+0x85c>  // b.any
    2af0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2af4:	a94573fb 	ldp	x27, x28, [sp, #80]
    2af8:	17ffff62 	b	2880 <sptLoadSparseTensor+0x17c>
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2afc:	52800300 	mov	w0, #0x18                  	// #24
	tsr->nnz = 0;
    2b00:	f9000e9f 	str	xzr, [x20, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2b04:	9ba07ec0 	umull	x0, w22, w0
    2b08:	97fff9d6 	bl	1260 <malloc@plt>
    2b0c:	f9001280 	str	x0, [x20, #32]
    2b10:	aa0003f9 	mov	x25, x0
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2b14:	b4002a80 	cbz	x0, 3064 <sptLoadSparseTensor+0x960>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2b18:	52800013 	mov	w19, #0x0                   	// #0
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2b1c:	5280031a 	mov	w26, #0x18                  	// #24
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2b20:	350000f6 	cbnz	w22, 2b3c <sptLoadSparseTensor+0x438>
    2b24:	140000b3 	b	2df0 <sptLoadSparseTensor+0x6ec>
    2b28:	b9400280 	ldr	w0, [x20]
    2b2c:	11000673 	add	w19, w19, #0x1
    2b30:	6b00027f 	cmp	w19, w0
    2b34:	540015e2 	b.cs	2df0 <sptLoadSparseTensor+0x6ec>  // b.hs, b.nlast
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2b38:	f9401299 	ldr	x25, [x20, #32]
    2b3c:	9bba6660 	umaddl	x0, w19, w26, x25
    2b40:	d2800002 	mov	x2, #0x0                   	// #0
    2b44:	d2800001 	mov	x1, #0x0                   	// #0
    2b48:	97fffdea 	bl	22f0 <sptNewIndexVector>
		spt_CheckError(retval, "SpTns Load", NULL);
    2b4c:	34fffee0 	cbz	w0, 2b28 <sptLoadSparseTensor+0x424>
    2b50:	2a0003e1 	mov	w1, w0
    2b54:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    2b58:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    2b5c:	910a6262 	add	x2, x19, #0x298
    2b60:	910b0280 	add	x0, x20, #0x2c0
    2b64:	d2800004 	mov	x4, #0x0                   	// #0
    2b68:	52800b43 	mov	w3, #0x5a                  	// #90
    2b6c:	940007a1 	bl	49f0 <spt_ComplainError>
    2b70:	a9446bf9 	ldp	x25, x26, [sp, #64]
			spt_CheckOSError(iores != 0, "SpTns Load");
    2b74:	97fffa27 	bl	1410 <__errno_location@plt>
    2b78:	b9400017 	ldr	w23, [x0]
    2b7c:	aa0003f6 	mov	x22, x0
    2b80:	314042ff 	cmn	w23, #0x10, lsl #12
    2b84:	54ffe7e0 	b.eq	2880 <sptLoadSparseTensor+0x17c>  // b.none
    2b88:	2a1703e0 	mov	w0, w23
    2b8c:	97fff9e1 	bl	1310 <strerror@plt>
    2b90:	910a6262 	add	x2, x19, #0x298
    2b94:	aa0003e4 	mov	x4, x0
    2b98:	114042e1 	add	w1, w23, #0x10, lsl #12
    2b9c:	910b0280 	add	x0, x20, #0x2c0
    2ba0:	52802483 	mov	w3, #0x124                 	// #292
    2ba4:	94000793 	bl	49f0 <spt_ComplainError>
    2ba8:	b94002c0 	ldr	w0, [x22]
    2bac:	11404000 	add	w0, w0, #0x10, lsl #12
    2bb0:	17ffff37 	b	288c <sptLoadSparseTensor+0x188>
	result = sptResizeValueVector(&tsr->values, nnz);
    2bb4:	9100a280 	add	x0, x20, #0x28
    2bb8:	97fffd97 	bl	2214 <sptResizeValueVector>
	spt_CheckError(result, "SpTns Read", NULL);
    2bbc:	35002800 	cbnz	w0, 30bc <sptLoadSparseTensor+0x9b8>
	for(sptIndex m=0; m < nmodes; ++m) {
    2bc0:	b94077e0 	ldr	w0, [sp, #116]
    2bc4:	b9006bff 	str	wzr, [sp, #104]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2bc8:	f9403ff3 	ldr	x19, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2bcc:	34002740 	cbz	w0, 30b4 <sptLoadSparseTensor+0x9b0>
    2bd0:	90000019 	adrp	x25, 2000 <sptFreeSparseTensor+0x50>
    2bd4:	910263fa 	add	x26, sp, #0x98
    2bd8:	91178339 	add	x25, x25, #0x5e0
    2bdc:	d503201f 	nop
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2be0:	b9406be2 	ldr	w2, [sp, #104]
    2be4:	52800303 	mov	w3, #0x18                  	// #24
    2be8:	f9401280 	ldr	x0, [x20, #32]
	if(header->idx_width == sizeof(sptIndex)) {
    2bec:	f94047e1 	ldr	x1, [sp, #136]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2bf0:	9ba30040 	umaddl	x0, w2, w3, x0
    2bf4:	f9400817 	ldr	x23, [x0, #16]
	if(header->idx_width == sizeof(sptIndex)) {
    2bf8:	f100103f 	cmp	x1, #0x4
    2bfc:	54000940 	b.eq	2d24 <sptLoadSparseTensor+0x620>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2c00:	d2a00800 	mov	x0, #0x400000              	// #4194304
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2c04:	2a1303fc 	mov	w28, w19
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2c08:	97fff996 	bl	1260 <malloc@plt>
    2c0c:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2c10:	34000313 	cbz	w19, 2c70 <sptLoadSparseTensor+0x56c>
    2c14:	5280001b 	mov	w27, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2c18:	52a00218 	mov	w24, #0x100000              	// #1048576
    2c1c:	d503201f 	nop
    2c20:	4b1b0393 	sub	w19, w28, w27
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2c24:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2c28:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2c2c:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2c30:	1a989273 	csel	w19, w19, w24, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2c34:	aa1603e0 	mov	x0, x22
    2c38:	2a1303e2 	mov	w2, w19
    2c3c:	97fff9d5 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    2c40:	a909dbf7 	stp	x23, x22, [sp, #152]
    2c44:	aa1a03e1 	mov	x1, x26
    2c48:	aa1903e0 	mov	x0, x25
    2c4c:	52800003 	mov	w3, #0x0                   	// #0
    2c50:	52800002 	mov	w2, #0x0                   	// #0
    2c54:	29154ffb 	stp	w27, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2c58:	1144037b 	add	w27, w27, #0x100, lsl #12
    2c5c:	97fff9f1 	bl	1420 <GOMP_parallel@plt>
    2c60:	6b1b039f 	cmp	w28, w27
    2c64:	54fffde8 	b.hi	2c20 <sptLoadSparseTensor+0x51c>  // b.pmore
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2c68:	f9403ff3 	ldr	x19, [sp, #120]
    2c6c:	2a1303fc 	mov	w28, w19
		free(ubuf);
    2c70:	aa1603e0 	mov	x0, x22
    2c74:	97fff9cb 	bl	13a0 <free@plt>
	for(sptIndex m=0; m < nmodes; ++m) {
    2c78:	b9406be1 	ldr	w1, [sp, #104]
    2c7c:	b94077e0 	ldr	w0, [sp, #116]
    2c80:	11000421 	add	w1, w1, #0x1
    2c84:	b9006be1 	str	w1, [sp, #104]
    2c88:	6b00003f 	cmp	w1, w0
    2c8c:	54fffaa3 	b.cc	2be0 <sptLoadSparseTensor+0x4dc>  // b.lo, b.ul, b.last
	if(header->val_width == sizeof(sptValue)) {
    2c90:	f9404be1 	ldr	x1, [sp, #144]
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2c94:	f9401e98 	ldr	x24, [x20, #56]
	if(header->val_width == sizeof(sptValue)) {
    2c98:	f100103f 	cmp	x1, #0x4
    2c9c:	54001820 	b.eq	2fa0 <sptLoadSparseTensor+0x89c>  // b.none
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    2ca0:	d2a01000 	mov	x0, #0x800000              	// #8388608
    2ca4:	90000017 	adrp	x23, 2000 <sptFreeSparseTensor+0x50>
    2ca8:	97fff96e 	bl	1260 <malloc@plt>
    2cac:	9119c2f7 	add	x23, x23, #0x670
    2cb0:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2cb4:	910263f9 	add	x25, sp, #0x98
    2cb8:	52800014 	mov	w20, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2cbc:	52a0021a 	mov	w26, #0x100000              	// #1048576
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2cc0:	3400029c 	cbz	w28, 2d10 <sptLoadSparseTensor+0x60c>
    2cc4:	d503201f 	nop
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2cc8:	4b140393 	sub	w19, w28, w20
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2ccc:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2cd0:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2cd4:	d2800101 	mov	x1, #0x8                   	// #8
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2cd8:	1a9a9273 	csel	w19, w19, w26, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2cdc:	aa1603e0 	mov	x0, x22
    2ce0:	2a1303e2 	mov	w2, w19
    2ce4:	97fff9ab 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    2ce8:	a909dbf8 	stp	x24, x22, [sp, #152]
    2cec:	aa1903e1 	mov	x1, x25
    2cf0:	aa1703e0 	mov	x0, x23
    2cf4:	52800003 	mov	w3, #0x0                   	// #0
    2cf8:	52800002 	mov	w2, #0x0                   	// #0
    2cfc:	29154ff4 	stp	w20, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2d00:	11440294 	add	w20, w20, #0x100, lsl #12
    2d04:	97fff9c7 	bl	1420 <GOMP_parallel@plt>
    2d08:	6b1c029f 	cmp	w20, w28
    2d0c:	54fffde3 	b.cc	2cc8 <sptLoadSparseTensor+0x5c4>  // b.lo, b.ul, b.last
		free(ubuf);
    2d10:	aa1603e0 	mov	x0, x22
    2d14:	97fff9a3 	bl	13a0 <free@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2d18:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2d1c:	a94573fb 	ldp	x27, x28, [sp, #80]
    2d20:	17fffed8 	b	2880 <sptLoadSparseTensor+0x17c>
		fread(buffer, sizeof(sptIndex), count, fin);
    2d24:	2a1303e2 	mov	w2, w19
    2d28:	aa1703e0 	mov	x0, x23
    2d2c:	aa1503e3 	mov	x3, x21
    2d30:	97fff998 	bl	1390 <fread@plt>
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2d34:	f9403ff3 	ldr	x19, [sp, #120]
    2d38:	2a1303fc 	mov	w28, w19
    2d3c:	17ffffcf 	b	2c78 <sptLoadSparseTensor+0x574>
		fread(buffer, sizeof(sptNnzIndex), count, fin);
    2d40:	aa1303e1 	mov	x1, x19
    2d44:	aa1503e3 	mov	x3, x21
    2d48:	9101e3e0 	add	x0, sp, #0x78
    2d4c:	d2800022 	mov	x2, #0x1                   	// #1
    2d50:	97fff990 	bl	1390 <fread@plt>
    2d54:	17ffff46 	b	2a6c <sptLoadSparseTensor+0x368>
		fread(buffer, sizeof(sptIndex), count, fin);
    2d58:	aa1503e3 	mov	x3, x21
    2d5c:	d2800022 	mov	x2, #0x1                   	// #1
    2d60:	9101d3e0 	add	x0, sp, #0x74
    2d64:	97fff98b 	bl	1390 <fread@plt>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    2d68:	b94077f7 	ldr	w23, [sp, #116]
    2d6c:	2a1703f6 	mov	w22, w23
    2d70:	d37e7ee0 	ubfiz	x0, x23, #2, #32
    2d74:	97fff93b 	bl	1260 <malloc@plt>
    2d78:	f94047f3 	ldr	x19, [sp, #136]
    2d7c:	aa0003fa 	mov	x26, x0
	if(header->idx_width == sizeof(sptIndex)) {
    2d80:	f100127f 	cmp	x19, #0x4
    2d84:	54ffe0c1 	b.ne	299c <sptLoadSparseTensor+0x298>  // b.any
		fread(buffer, sizeof(sptIndex), count, fin);
    2d88:	aa1303e1 	mov	x1, x19
    2d8c:	aa1603e2 	mov	x2, x22
    2d90:	aa1503e3 	mov	x3, x21
    2d94:	97fff97f 	bl	1390 <fread@plt>
    2d98:	f94047f3 	ldr	x19, [sp, #136]
    2d9c:	17ffff1e 	b	2a14 <sptLoadSparseTensor+0x310>
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    2da0:	900000a1 	adrp	x1, 16000 <__FRAME_END__+0xfb88>
    2da4:	d37df003 	lsl	x3, x0, #3
    2da8:	aa0303e2 	mov	x2, x3
    2dac:	f947d820 	ldr	x0, [x1, #4016]
    2db0:	f0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    2db4:	910d0021 	add	x1, x1, #0x340
    2db8:	f9400000 	ldr	x0, [x0]
    2dbc:	97fff9a1 	bl	1440 <fprintf@plt>
    2dc0:	f94047e1 	ldr	x1, [sp, #136]
    2dc4:	17fffed6 	b	291c <sptLoadSparseTensor+0x218>
	tsr->nnz = 0;
    2dc8:	f9000e9f 	str	xzr, [x20, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2dcc:	d2800000 	mov	x0, #0x0                   	// #0
    2dd0:	97fff924 	bl	1260 <malloc@plt>
    2dd4:	f9001280 	str	x0, [x20, #32]
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2dd8:	b50000c0 	cbnz	x0, 2df0 <sptLoadSparseTensor+0x6ec>
    2ddc:	97fff98d 	bl	1410 <__errno_location@plt>
    2de0:	b9400013 	ldr	w19, [x0]
    2de4:	aa0003fa 	mov	x26, x0
    2de8:	3140427f 	cmn	w19, #0x10, lsl #12
    2dec:	54001461 	b.ne	3078 <sptLoadSparseTensor+0x974>  // b.any
	retval = sptNewValueVector(&tsr->values, 0, 0);
    2df0:	9100a29a 	add	x26, x20, #0x28
    2df4:	d2800002 	mov	x2, #0x0                   	// #0
    2df8:	aa1a03e0 	mov	x0, x26
    2dfc:	d2800001 	mov	x1, #0x0                   	// #0
    2e00:	910263f9 	add	x25, sp, #0x98
    2e04:	97fffc8b 	bl	2030 <sptNewValueVector>
	spt_CheckError(retval, "SpTns Load", NULL);
    2e08:	350016c0 	cbnz	w0, 30e0 <sptLoadSparseTensor+0x9dc>
			iores = fscanf(fp, "%u", &index);
    2e0c:	913de318 	add	x24, x24, #0xf78
    2e10:	a90573fb 	stp	x27, x28, [sp, #80]
			iores = fscanf(fp, "%lf", &value);
    2e14:	f000001b 	adrp	x27, 5000 <_IO_stdin_used+0x2f0>
    2e18:	910ba37b 	add	x27, x27, #0x2e8
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2e1c:	52800316 	mov	w22, #0x18                  	// #24
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e20:	b9400280 	ldr	w0, [x20]
    2e24:	52800013 	mov	w19, #0x0                   	// #0
    2e28:	350001a0 	cbnz	w0, 2e5c <sptLoadSparseTensor+0x758>
    2e2c:	14000020 	b	2eac <sptLoadSparseTensor+0x7a8>
			if(index < start_index) {
    2e30:	b9409be1 	ldr	w1, [sp, #152]
    2e34:	6b0102ff 	cmp	w23, w1
    2e38:	54000568 	b.hi	2ee4 <sptLoadSparseTensor+0x7e0>  // b.pmore
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2e3c:	f9401280 	ldr	x0, [x20, #32]
    2e40:	4b170021 	sub	w1, w1, w23
    2e44:	9bb60260 	umaddl	x0, w19, w22, x0
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e48:	11000673 	add	w19, w19, #0x1
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2e4c:	97fffd51 	bl	2390 <sptAppendIndexVector>
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e50:	b9400280 	ldr	w0, [x20]
    2e54:	6b00027f 	cmp	w19, w0
    2e58:	540002a2 	b.cs	2eac <sptLoadSparseTensor+0x7a8>  // b.hs, b.nlast
			iores = fscanf(fp, "%u", &index);
    2e5c:	aa1903e2 	mov	x2, x25
    2e60:	aa1803e1 	mov	x1, x24
    2e64:	aa1503e0 	mov	x0, x21
    2e68:	97fff902 	bl	1270 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2e6c:	7100041f 	cmp	w0, #0x1
    2e70:	54fffe00 	b.eq	2e30 <sptLoadSparseTensor+0x72c>  // b.none
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e74:	b9400280 	ldr	w0, [x20]
    2e78:	34ffe3c0 	cbz	w0, 2af0 <sptLoadSparseTensor+0x3ec>
    2e7c:	a9418a83 	ldp	x3, x2, [x20, #24]
    2e80:	51000400 	sub	w0, w0, #0x1
    2e84:	52800304 	mov	w4, #0x18                  	// #24
    2e88:	91006041 	add	x1, x2, #0x18
    2e8c:	9ba40400 	umaddl	x0, w0, w4, x1
    2e90:	14000002 	b	2e98 <sptLoadSparseTensor+0x794>
    2e94:	91006021 	add	x1, x1, #0x18
		tsr->inds[mode].len = tsr->nnz;
    2e98:	f9000043 	str	x3, [x2]
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e9c:	aa0103e2 	mov	x2, x1
    2ea0:	eb01001f 	cmp	x0, x1
    2ea4:	54ffff81 	b.ne	2e94 <sptLoadSparseTensor+0x790>  // b.any
    2ea8:	17ffff12 	b	2af0 <sptLoadSparseTensor+0x3ec>
			iores = fscanf(fp, "%lf", &value);
    2eac:	aa1903e2 	mov	x2, x25
    2eb0:	aa1b03e1 	mov	x1, x27
    2eb4:	aa1503e0 	mov	x0, x21
    2eb8:	97fff8ee 	bl	1270 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2ebc:	7100041f 	cmp	w0, #0x1
    2ec0:	54fffda1 	b.ne	2e74 <sptLoadSparseTensor+0x770>  // b.any
			sptAppendValueVector(&tsr->values, value);
    2ec4:	fd404fe0 	ldr	d0, [sp, #152]
    2ec8:	aa1a03e0 	mov	x0, x26
    2ecc:	1e624000 	fcvt	s0, d0
    2ed0:	97fffc9d 	bl	2144 <sptAppendValueVector>
			++tsr->nnz;
    2ed4:	f9400e80 	ldr	x0, [x20, #24]
    2ed8:	91000400 	add	x0, x0, #0x1
    2edc:	f9000e80 	str	x0, [x20, #24]
	while(retval == 0) {
    2ee0:	17ffffd0 	b	2e20 <sptLoadSparseTensor+0x71c>
				spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
    2ee4:	f0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    2ee8:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    2eec:	910b4084 	add	x4, x4, #0x2d0
    2ef0:	910a6262 	add	x2, x19, #0x298
    2ef4:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    2ef8:	52800d03 	mov	w3, #0x68                  	// #104
    2efc:	910b0280 	add	x0, x20, #0x2c0
    2f00:	52800061 	mov	w1, #0x3                   	// #3
    2f04:	940006bb 	bl	49f0 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2f08:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2f0c:	a94573fb 	ldp	x27, x28, [sp, #80]
    2f10:	17ffff19 	b	2b74 <sptLoadSparseTensor+0x470>
	spt_CheckOSError(iores < 0, "SpTns Load");
    2f14:	97fff93f 	bl	1410 <__errno_location@plt>
    2f18:	b9400013 	ldr	w19, [x0]
    2f1c:	aa0003f6 	mov	x22, x0
    2f20:	3140427f 	cmn	w19, #0x10, lsl #12
    2f24:	54ffc4e0 	b.eq	27c0 <sptLoadSparseTensor+0xbc>  // b.none
    2f28:	2a1303e0 	mov	w0, w19
    2f2c:	97fff8f9 	bl	1310 <strerror@plt>
    2f30:	11404261 	add	w1, w19, #0x10, lsl #12
    2f34:	aa0003e4 	mov	x4, x0
    2f38:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    2f3c:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    2f40:	910a6262 	add	x2, x19, #0x298
    2f44:	910b0280 	add	x0, x20, #0x2c0
    2f48:	52800943 	mov	w3, #0x4a                  	// #74
    2f4c:	940006a9 	bl	49f0 <spt_ComplainError>
    2f50:	b94002c0 	ldr	w0, [x22]
    2f54:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2f58:	34ffc940 	cbz	w0, 2880 <sptLoadSparseTensor+0x17c>
    2f5c:	17ffff06 	b	2b74 <sptLoadSparseTensor+0x470>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2f60:	2a1603e0 	mov	w0, w22
    2f64:	97fff8eb 	bl	1310 <strerror@plt>
    2f68:	910a6262 	add	x2, x19, #0x298
    2f6c:	aa0003e4 	mov	x4, x0
    2f70:	114042c1 	add	w1, w22, #0x10, lsl #12
    2f74:	f0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    2f78:	52802503 	mov	w3, #0x128                 	// #296
    2f7c:	910b0000 	add	x0, x0, #0x2c0
    2f80:	9400069c 	bl	49f0 <spt_ComplainError>
    2f84:	b9400280 	ldr	w0, [x20]
    2f88:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2f8c:	11404000 	add	w0, w0, #0x10, lsl #12
    2f90:	a94573fb 	ldp	x27, x28, [sp, #80]
    2f94:	17fffe3e 	b	288c <sptLoadSparseTensor+0x188>
	} while(file_extensions[++idx].extension != NULL);
    2f98:	d2800021 	mov	x1, #0x1                   	// #1
    2f9c:	17fffe42 	b	28a4 <sptLoadSparseTensor+0x1a0>
		fread(buffer, sizeof(sptValue), count, fin);
    2fa0:	2a1303e2 	mov	w2, w19
    2fa4:	aa1803e0 	mov	x0, x24
    2fa8:	aa1503e3 	mov	x3, x21
    2fac:	97fff8f9 	bl	1390 <fread@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2fb0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2fb4:	a94573fb 	ldp	x27, x28, [sp, #80]
    2fb8:	17fffe32 	b	2880 <sptLoadSparseTensor+0x17c>
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    2fbc:	97fff915 	bl	1410 <__errno_location@plt>
    2fc0:	b940001a 	ldr	w26, [x0]
    2fc4:	aa0003f9 	mov	x25, x0
    2fc8:	3140435f 	cmn	w26, #0x10, lsl #12
    2fcc:	54ffc0a0 	b.eq	27e0 <sptLoadSparseTensor+0xdc>  // b.none
    2fd0:	2a1a03e0 	mov	w0, w26
    2fd4:	97fff8cf 	bl	1310 <strerror@plt>
    2fd8:	11404341 	add	w1, w26, #0x10, lsl #12
    2fdc:	aa0003e4 	mov	x4, x0
    2fe0:	f0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    2fe4:	f0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    2fe8:	910a6262 	add	x2, x19, #0x298
    2fec:	910b0280 	add	x0, x20, #0x2c0
    2ff0:	528009a3 	mov	w3, #0x4d                  	// #77
    2ff4:	9400067f 	bl	49f0 <spt_ComplainError>
    2ff8:	b9400320 	ldr	w0, [x25]
    2ffc:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3000:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    3004:	34ffc3e0 	cbz	w0, 2880 <sptLoadSparseTensor+0x17c>
    3008:	17fffedb 	b	2b74 <sptLoadSparseTensor+0x470>
    300c:	f90037e0 	str	x0, [sp, #104]
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    3010:	97fff900 	bl	1410 <__errno_location@plt>
    3014:	b9400013 	ldr	w19, [x0]
    3018:	aa0003f9 	mov	x25, x0
    301c:	f94037e2 	ldr	x2, [sp, #104]
    3020:	3140427f 	cmn	w19, #0x10, lsl #12
    3024:	54ffbe80 	b.eq	27f4 <sptLoadSparseTensor+0xf0>  // b.none
    3028:	2a1303e0 	mov	w0, w19
    302c:	97fff8b9 	bl	1310 <strerror@plt>
    3030:	11404261 	add	w1, w19, #0x10, lsl #12
    3034:	aa0003e4 	mov	x4, x0
    3038:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    303c:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    3040:	910a6262 	add	x2, x19, #0x298
    3044:	910b0280 	add	x0, x20, #0x2c0
    3048:	52800a03 	mov	w3, #0x50                  	// #80
    304c:	94000669 	bl	49f0 <spt_ComplainError>
    3050:	b9400320 	ldr	w0, [x25]
    3054:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3058:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    305c:	34ffc120 	cbz	w0, 2880 <sptLoadSparseTensor+0x17c>
    3060:	17fffec5 	b	2b74 <sptLoadSparseTensor+0x470>
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    3064:	97fff8eb 	bl	1410 <__errno_location@plt>
    3068:	b9400013 	ldr	w19, [x0]
    306c:	aa0003fa 	mov	x26, x0
    3070:	3140427f 	cmn	w19, #0x10, lsl #12
    3074:	54ffd520 	b.eq	2b18 <sptLoadSparseTensor+0x414>  // b.none
    3078:	2a1303e0 	mov	w0, w19
    307c:	97fff8a5 	bl	1310 <strerror@plt>
    3080:	11404261 	add	w1, w19, #0x10, lsl #12
    3084:	aa0003e4 	mov	x4, x0
    3088:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    308c:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    3090:	910a6262 	add	x2, x19, #0x298
    3094:	910b0280 	add	x0, x20, #0x2c0
    3098:	52800ae3 	mov	w3, #0x57                  	// #87
    309c:	94000655 	bl	49f0 <spt_ComplainError>
    30a0:	b9400340 	ldr	w0, [x26]
    30a4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    30a8:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    30ac:	34ffbea0 	cbz	w0, 2880 <sptLoadSparseTensor+0x17c>
    30b0:	17fffeb1 	b	2b74 <sptLoadSparseTensor+0x470>
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    30b4:	2a1303fc 	mov	w28, w19
    30b8:	17fffef6 	b	2c90 <sptLoadSparseTensor+0x58c>
	spt_CheckError(result, "SpTns Read", NULL);
    30bc:	2a0003e1 	mov	w1, w0
    30c0:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    30c4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    30c8:	910a6262 	add	x2, x19, #0x298
    30cc:	910ee000 	add	x0, x0, #0x3b8
    30d0:	d2800004 	mov	x4, #0x0                   	// #0
    30d4:	52802103 	mov	w3, #0x108                 	// #264
    30d8:	94000646 	bl	49f0 <spt_ComplainError>
    30dc:	17fffe80 	b	2adc <sptLoadSparseTensor+0x3d8>
	spt_CheckError(retval, "SpTns Load", NULL);
    30e0:	2a0003e1 	mov	w1, w0
    30e4:	d0000013 	adrp	x19, 5000 <_IO_stdin_used+0x2f0>
    30e8:	d0000014 	adrp	x20, 5000 <_IO_stdin_used+0x2f0>
    30ec:	910a6262 	add	x2, x19, #0x298
    30f0:	910b0280 	add	x0, x20, #0x2c0
    30f4:	d2800004 	mov	x4, #0x0                   	// #0
    30f8:	52800ba3 	mov	w3, #0x5d                  	// #93
    30fc:	9400063d 	bl	49f0 <spt_ComplainError>
    3100:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3104:	17fffe9c 	b	2b74 <sptLoadSparseTensor+0x470>
	sptAssert(fp != NULL);
    3108:	12800000 	mov	w0, #0xffffffff            	// #-1
    310c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3110:	a90573fb 	stp	x27, x28, [sp, #80]
    3114:	97fff82b 	bl	11c0 <exit@plt>
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    3118:	f0000080 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    311c:	d37df023 	lsl	x3, x1, #3
    3120:	aa0303e2 	mov	x2, x3
    3124:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    3128:	f947d800 	ldr	x0, [x0, #4016]
    312c:	910bc021 	add	x1, x1, #0x2f0
    3130:	f9400000 	ldr	x0, [x0]
    3134:	97fff8c3 	bl	1440 <fprintf@plt>
		exit(-1);
    3138:	12800000 	mov	w0, #0xffffffff            	// #-1
    313c:	97fff821 	bl	11c0 <exit@plt>

0000000000003140 <sptNewMatrix>:
 */
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
	mtx->nrows = nrows;
	mtx->ncols = ncols;
	mtx->cap = nrows != 0 ? nrows : 1;
	mtx->stride = ((ncols-1)/8+1)*8;
    3140:	51000443 	sub	w3, w2, #0x1
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3144:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    3148:	7100003f 	cmp	w1, #0x0
	mtx->stride = ((ncols-1)/8+1)*8;
    314c:	121d7063 	and	w3, w3, #0xfffffff8
    3150:	1a9f1424 	csinc	w4, w1, wzr, ne  // ne = any
    3154:	11002063 	add	w3, w3, #0x8
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3158:	910003fd 	mov	x29, sp
    315c:	a90153f3 	stp	x19, x20, [sp, #16]
    3160:	aa0003f3 	mov	x19, x0
#ifdef _ISOC11_SOURCE
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    3164:	1b047c74 	mul	w20, w3, w4
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3168:	a9025bf5 	stp	x21, x22, [sp, #32]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    316c:	d2800400 	mov	x0, #0x20                  	// #32
	mtx->ncols = ncols;
    3170:	29000a61 	stp	w1, w2, [x19]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    3174:	d37ef694 	lsl	x20, x20, #2
	mtx->stride = ((ncols-1)/8+1)*8;
    3178:	29010e64 	stp	w4, w3, [x19, #8]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    317c:	aa1403e1 	mov	x1, x20
    3180:	97fff848 	bl	12a0 <aligned_alloc@plt>
    3184:	f9000a60 	str	x0, [x19, #16]
    3188:	aa0003f5 	mov	x21, x0
		}
	}
#else
	mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
	spt_CheckOSError(!mtx->values, "Mtx New");
    318c:	b4000140 	cbz	x0, 31b4 <sptNewMatrix+0x74>
	memset(mtx->values, 0, mtx->cap * mtx->stride * sizeof (sptValue));
    3190:	aa1403e2 	mov	x2, x20
    3194:	aa1503e0 	mov	x0, x21
    3198:	52800001 	mov	w1, #0x0                   	// #0
    319c:	97fff845 	bl	12b0 <memset@plt>
	return 0;
    31a0:	52800000 	mov	w0, #0x0                   	// #0
}
    31a4:	a94153f3 	ldp	x19, x20, [sp, #16]
    31a8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    31ac:	a8c37bfd 	ldp	x29, x30, [sp], #48
    31b0:	d65f03c0 	ret
	spt_CheckOSError(!mtx->values, "Mtx New");
    31b4:	97fff897 	bl	1410 <__errno_location@plt>
    31b8:	b9400016 	ldr	w22, [x0]
    31bc:	aa0003f3 	mov	x19, x0
    31c0:	314042df 	cmn	w22, #0x10, lsl #12
    31c4:	54fffe60 	b.eq	3190 <sptNewMatrix+0x50>  // b.none
    31c8:	2a1603e0 	mov	w0, w22
    31cc:	97fff851 	bl	1310 <strerror@plt>
    31d0:	114042c1 	add	w1, w22, #0x10, lsl #12
    31d4:	aa0003e4 	mov	x4, x0
    31d8:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    31dc:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    31e0:	910f4042 	add	x2, x2, #0x3d0
    31e4:	910fe000 	add	x0, x0, #0x3f8
    31e8:	52800703 	mov	w3, #0x38                  	// #56
    31ec:	94000601 	bl	49f0 <spt_ComplainError>
    31f0:	b9400260 	ldr	w0, [x19]
    31f4:	11404000 	add	w0, w0, #0x10, lsl #12
    31f8:	17ffffeb 	b	31a4 <sptNewMatrix+0x64>
    31fc:	d503201f 	nop

0000000000003200 <sptRandomizeMatrix>:
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, bool random) {
    3200:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    3204:	910003fd 	mov	x29, sp
    3208:	a90363f7 	stp	x23, x24, [sp, #48]
    320c:	aa0003f8 	mov	x24, x0
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3210:	b9400000 	ldr	w0, [x0]
    3214:	34000440 	cbz	w0, 329c <sptRandomizeMatrix+0x9c>
    3218:	12001c37 	and	w23, w1, #0xff
    321c:	a9025bf5 	stp	x21, x22, [sp, #32]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
			if (random) {
				srand(time(NULL) + (rand()%100) + i + j);
    3220:	5290a3f6 	mov	w22, #0x851f                	// #34079
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3224:	b9400700 	ldr	w0, [x24, #4]
				srand(time(NULL) + (rand()%100) + i + j);
    3228:	72aa3d76 	movk	w22, #0x51eb, lsl #16
    322c:	52800c95 	mov	w21, #0x64                  	// #100
    3230:	a90153f3 	stp	x19, x20, [sp, #16]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3234:	52800014 	mov	w20, #0x0                   	// #0
    3238:	a9046bf9 	stp	x25, x26, [sp, #64]
    323c:	f9002bfb 	str	x27, [sp, #80]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3240:	34000260 	cbz	w0, 328c <sptRandomizeMatrix+0x8c>
    3244:	35000357 	cbnz	w23, 32ac <sptRandomizeMatrix+0xac>
    3248:	11134a9a 	add	w26, w20, #0x4d2
    324c:	52800013 	mov	w19, #0x0                   	// #0
			} else {
				srand(1234 + i + j);
    3250:	0b130340 	add	w0, w26, w19
    3254:	97fff863 	bl	13e0 <srand@plt>
			}
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3258:	b9400f19 	ldr	w25, [x24, #12]
    325c:	f9400b1b 	ldr	x27, [x24, #16]
    3260:	94000664 	bl	4bf0 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3264:	b9400700 	ldr	w0, [x24, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3268:	1b194e99 	madd	w25, w20, w25, w19
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    326c:	11000673 	add	w19, w19, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3270:	bc397b60 	str	s0, [x27, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3274:	6b00027f 	cmp	w19, w0
    3278:	54fffec3 	b.cc	3250 <sptRandomizeMatrix+0x50>  // b.lo, b.ul, b.last
	for(sptIndex i=0; i<mtx->nrows; ++i)
    327c:	b9400301 	ldr	w1, [x24]
    3280:	11000694 	add	w20, w20, #0x1
    3284:	6b01029f 	cmp	w20, w1
    3288:	54fffdc3 	b.cc	3240 <sptRandomizeMatrix+0x40>  // b.lo, b.ul, b.last
    328c:	a94153f3 	ldp	x19, x20, [sp, #16]
    3290:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3294:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3298:	f9402bfb 	ldr	x27, [sp, #80]
		}
	return 0;
}
    329c:	52800000 	mov	w0, #0x0                   	// #0
    32a0:	a94363f7 	ldp	x23, x24, [sp, #48]
    32a4:	a8c67bfd 	ldp	x29, x30, [sp], #96
    32a8:	d65f03c0 	ret
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32ac:	52800013 	mov	w19, #0x0                   	// #0
				srand(time(NULL) + (rand()%100) + i + j);
    32b0:	d2800000 	mov	x0, #0x0                   	// #0
    32b4:	97fff7e7 	bl	1250 <time@plt>
    32b8:	aa0003f9 	mov	x25, x0
    32bc:	97fff809 	bl	12e0 <rand@plt>
    32c0:	9b367c01 	smull	x1, w0, w22
    32c4:	0b13029a 	add	w26, w20, w19
    32c8:	9365fc21 	asr	x1, x1, #37
    32cc:	4b807c21 	sub	w1, w1, w0, asr #31
    32d0:	1b158021 	msub	w1, w1, w21, w0
    32d4:	0b190020 	add	w0, w1, w25
    32d8:	0b1a0000 	add	w0, w0, w26
    32dc:	97fff841 	bl	13e0 <srand@plt>
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    32e0:	b9400f19 	ldr	w25, [x24, #12]
    32e4:	f9400b1a 	ldr	x26, [x24, #16]
    32e8:	94000642 	bl	4bf0 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32ec:	b9400700 	ldr	w0, [x24, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    32f0:	1b194e99 	madd	w25, w20, w25, w19
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32f4:	11000673 	add	w19, w19, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    32f8:	bc397b40 	str	s0, [x26, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    32fc:	6b13001f 	cmp	w0, w19
    3300:	54fffd88 	b.hi	32b0 <sptRandomizeMatrix+0xb0>  // b.pmore
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3304:	b9400301 	ldr	w1, [x24]
    3308:	11000694 	add	w20, w20, #0x1
    330c:	6b01029f 	cmp	w20, w1
    3310:	54fff983 	b.cc	3240 <sptRandomizeMatrix+0x40>  // b.lo, b.ul, b.last
    3314:	17ffffde 	b	328c <sptRandomizeMatrix+0x8c>
    3318:	d503201f 	nop
    331c:	d503201f 	nop

0000000000003320 <sptConstantMatrix>:
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3320:	b9400004 	ldr	w4, [x0]
    3324:	34000204 	cbz	w4, 3364 <sptConstantMatrix+0x44>
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3328:	b9400405 	ldr	w5, [x0, #4]
    332c:	340001c5 	cbz	w5, 3364 <sptConstantMatrix+0x44>
			mtx->values[i * mtx->stride + j] = val;
    3330:	b9400c06 	ldr	w6, [x0, #12]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3334:	52800003 	mov	w3, #0x0                   	// #0
			mtx->values[i * mtx->stride + j] = val;
    3338:	f9400802 	ldr	x2, [x0, #16]
    333c:	d503201f 	nop
    3340:	1b037cc0 	mul	w0, w6, w3
    3344:	0b050001 	add	w1, w0, w5
    3348:	bc205840 	str	s0, [x2, w0, uxtw #2]
		for(sptIndex j=0; j<mtx->ncols; ++j)
    334c:	11000400 	add	w0, w0, #0x1
    3350:	6b00003f 	cmp	w1, w0
    3354:	54ffffa1 	b.ne	3348 <sptConstantMatrix+0x28>  // b.any
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3358:	11000463 	add	w3, w3, #0x1
    335c:	6b04007f 	cmp	w3, w4
    3360:	54ffff01 	b.ne	3340 <sptConstantMatrix+0x20>  // b.any
	return 0;
}
    3364:	52800000 	mov	w0, #0x0                   	// #0
    3368:	d65f03c0 	ret
    336c:	d503201f 	nop

0000000000003370 <sptFreeMatrix>:
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    3370:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    3374:	910003fd 	mov	x29, sp
    3378:	f9000bf3 	str	x19, [sp, #16]
    337c:	aa0003f3 	mov	x19, x0
	free(mtx->values);
    3380:	f9400800 	ldr	x0, [x0, #16]
    3384:	97fff807 	bl	13a0 <free@plt>
	mtx->nrows = 0;
    3388:	a9007e7f 	stp	xzr, xzr, [x19]
	mtx->ncols = 0;
	mtx->cap = 0;
	mtx->stride = 0;
}
    338c:	f9400bf3 	ldr	x19, [sp, #16]
    3390:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3394:	d65f03c0 	ret
    3398:	d503201f 	nop
    339c:	d503201f 	nop

00000000000033a0 <sptSparseTensorDensity>:
#include <bits/types/FILE.h>
#include <stdio.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
    33a0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    33a4:	910003fd 	mov	x29, sp
    33a8:	f9000bf3 	str	x19, [sp, #16]
    33ac:	aa0003f3 	mov	x19, x0
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    33b0:	b9400000 	ldr	w0, [x0]
{
    33b4:	fd000fe8 	str	d8, [sp, #24]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    33b8:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
    33bc:	fd400e60 	ldr	d0, [x19, #24]
    33c0:	1e630001 	ucvtf	d1, w0
    33c4:	7e61d800 	ucvtf	d0, d0
    33c8:	1e611901 	fdiv	d1, d8, d1
    33cc:	97fff791 	bl	1210 <pow@plt>
	double density = 1.0;
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    33d0:	b9400262 	ldr	w2, [x19]
    33d4:	34000202 	cbz	w2, 3414 <sptSparseTensorDensity+0x74>
    33d8:	1e604002 	fmov	d2, d0
	double density = 1.0;
    33dc:	1e604100 	fmov	d0, d8
		density *= root / (double)tsr->ndims[m];
    33e0:	f9400a63 	ldr	x3, [x19, #16]
    33e4:	d2800000 	mov	x0, #0x0                   	// #0
    33e8:	b8607861 	ldr	w1, [x3, x0, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    33ec:	91000400 	add	x0, x0, #0x1
		density *= root / (double)tsr->ndims[m];
    33f0:	1e630021 	ucvtf	d1, w1
    33f4:	1e611841 	fdiv	d1, d2, d1
    33f8:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    33fc:	6b00005f 	cmp	w2, w0
    3400:	54ffff48 	b.hi	33e8 <sptSparseTensorDensity+0x48>  // b.pmore
	}

	return density;
}
    3404:	f9400bf3 	ldr	x19, [sp, #16]
    3408:	fd400fe8 	ldr	d8, [sp, #24]
    340c:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3410:	d65f03c0 	ret
	double density = 1.0;
    3414:	1e604100 	fmov	d0, d8
}
    3418:	fd400fe8 	ldr	d8, [sp, #24]
    341c:	f9400bf3 	ldr	x19, [sp, #16]
    3420:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3424:	d65f03c0 	ret
    3428:	d503201f 	nop
    342c:	d503201f 	nop

0000000000003430 <sptSparseTensorStatus>:



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
    3430:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3434:	aa0103e3 	mov	x3, x1
    3438:	d2800822 	mov	x2, #0x41                  	// #65
{
    343c:	910003fd 	mov	x29, sp
    3440:	a90153f3 	stp	x19, x20, [sp, #16]
    3444:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3448:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    344c:	91100000 	add	x0, x0, #0x400
{
    3450:	a9025bf5 	stp	x21, x22, [sp, #32]
    3454:	aa0103f5 	mov	x21, x1
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3458:	d2800021 	mov	x1, #0x1                   	// #1
{
    345c:	fd001be8 	str	d8, [sp, #48]
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3460:	97fff7d4 	bl	13b0 <fwrite@plt>
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3464:	d0000016 	adrp	x22, 5000 <_IO_stdin_used+0x2f0>
    3468:	f9400a62 	ldr	x2, [x19, #16]
    346c:	aa1503e0 	mov	x0, x21
    3470:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    3474:	91112021 	add	x1, x1, #0x448
    3478:	9111a2d6 	add	x22, x22, #0x468
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    347c:	52800034 	mov	w20, #0x1                   	// #1
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3480:	b9400042 	ldr	w2, [x2]
    3484:	97fff7ef 	bl	1440 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3488:	b9400260 	ldr	w0, [x19]
    348c:	6b14001f 	cmp	w0, w20
    3490:	54000169 	b.ls	34bc <sptSparseTensorStatus+0x8c>  // b.plast
    3494:	d503201f 	nop
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    3498:	f9400a62 	ldr	x2, [x19, #16]
    349c:	aa1603e1 	mov	x1, x22
    34a0:	aa1503e0 	mov	x0, x21
    34a4:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    34a8:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    34ac:	97fff7e5 	bl	1440 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    34b0:	b9400260 	ldr	w0, [x19]
    34b4:	6b14001f 	cmp	w0, w20
    34b8:	54ffff08 	b.hi	3498 <sptSparseTensorStatus+0x68>  // b.pmore
	}
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    34bc:	f9400e62 	ldr	x2, [x19, #24]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    34c0:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    34c4:	aa1503e0 	mov	x0, x21
    34c8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    34cc:	91116021 	add	x1, x1, #0x458
    34d0:	97fff7dc 	bl	1440 <fprintf@plt>
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    34d4:	b9400260 	ldr	w0, [x19]
    34d8:	fd400e60 	ldr	d0, [x19, #24]
    34dc:	1e630001 	ucvtf	d1, w0
    34e0:	7e61d800 	ucvtf	d0, d0
    34e4:	1e611901 	fdiv	d1, d8, d1
    34e8:	97fff74a 	bl	1210 <pow@plt>
    34ec:	1e604002 	fmov	d2, d0
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34f0:	b9400261 	ldr	w1, [x19]
	double density = 1.0;
    34f4:	1e604100 	fmov	d0, d8
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    34f8:	34000161 	cbz	w1, 3524 <sptSparseTensorStatus+0xf4>
		density *= root / (double)tsr->ndims[m];
    34fc:	f9400a63 	ldr	x3, [x19, #16]
    3500:	d2800002 	mov	x2, #0x0                   	// #0
    3504:	d503201f 	nop
    3508:	b8627860 	ldr	w0, [x3, x2, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    350c:	91000442 	add	x2, x2, #0x1
		density *= root / (double)tsr->ndims[m];
    3510:	1e630001 	ucvtf	d1, w0
    3514:	1e611841 	fdiv	d1, d2, d1
    3518:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    351c:	6b02003f 	cmp	w1, w2
    3520:	54ffff48 	b.hi	3508 <sptSparseTensorStatus+0xd8>  // b.pmore
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));
    3524:	aa1503e0 	mov	x0, x21
    3528:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    352c:	9111c021 	add	x1, x1, #0x470
    3530:	97fff7c4 	bl	1440 <fprintf@plt>

	fprintf(fp, "Average slice length (c): ");
    3534:	aa1503e3 	mov	x3, x21
    3538:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    353c:	d2800342 	mov	x2, #0x1a                  	// #26
    3540:	91120000 	add	x0, x0, #0x480
    3544:	d2800021 	mov	x1, #0x1                   	// #1
    3548:	97fff79a 	bl	13b0 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    354c:	b9400260 	ldr	w0, [x19]
    3550:	7100041f 	cmp	w0, #0x1
    3554:	54000640 	b.eq	361c <sptSparseTensorStatus+0x1ec>  // b.none
    3558:	d0000016 	adrp	x22, 5000 <_IO_stdin_used+0x2f0>
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    355c:	911282d6 	add	x22, x22, #0x4a0
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3560:	52800014 	mov	w20, #0x0                   	// #0
    3564:	d503201f 	nop
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3568:	f9400a62 	ldr	x2, [x19, #16]
    356c:	aa1603e1 	mov	x1, x22
    3570:	fd400e60 	ldr	d0, [x19, #24]
    3574:	aa1503e0 	mov	x0, x21
    3578:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    357c:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3580:	7e61d800 	ucvtf	d0, d0
    3584:	1e630041 	ucvtf	d1, w2
    3588:	1e611800 	fdiv	d0, d0, d1
    358c:	97fff7ad 	bl	1440 <fprintf@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3590:	b9400262 	ldr	w2, [x19]
    3594:	51000442 	sub	w2, w2, #0x1
    3598:	6b14005f 	cmp	w2, w20
    359c:	54fffe68 	b.hi	3568 <sptSparseTensorStatus+0x138>  // b.pmore
	}
	fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);
    35a0:	d37e7c42 	ubfiz	x2, x2, #2, #32
    35a4:	f9400a60 	ldr	x0, [x19, #16]
    35a8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    35ac:	fd400e60 	ldr	d0, [x19, #24]
    35b0:	9112c021 	add	x1, x1, #0x4b0
    35b4:	b8626802 	ldr	w2, [x0, x2]
    35b8:	aa1503e0 	mov	x0, x21
    35bc:	7e61d800 	ucvtf	d0, d0
    35c0:	1e630041 	ucvtf	d1, w2
    35c4:	1e611800 	fdiv	d0, d0, d1
    35c8:	97fff79e 	bl	1440 <fprintf@plt>

	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    35cc:	b9400261 	ldr	w1, [x19]
    35d0:	f9400e60 	ldr	x0, [x19, #24]
    35d4:	9b000020 	madd	x0, x1, x0, x0
    35d8:	d37ef400 	lsl	x0, x0, #2
    35dc:	94000541 	bl	4ae0 <sptBytesString>
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    35e0:	aa0003e2 	mov	x2, x0
	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    35e4:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    35e8:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    35ec:	aa1503e0 	mov	x0, x21
    35f0:	9112e021 	add	x1, x1, #0x4b8
    35f4:	97fff793 	bl	1440 <fprintf@plt>
	fprintf(fp, "\n");
    35f8:	aa1503e1 	mov	x1, x21
    35fc:	52800140 	mov	w0, #0xa                   	// #10
    3600:	97fff6f8 	bl	11e0 <fputc@plt>
	free(bytestr);
    3604:	aa1303e0 	mov	x0, x19
}
    3608:	fd401be8 	ldr	d8, [sp, #48]
    360c:	a94153f3 	ldp	x19, x20, [sp, #16]
    3610:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3614:	a8c47bfd 	ldp	x29, x30, [sp], #64
	free(bytestr);
    3618:	17fff762 	b	13a0 <free@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    361c:	d2800002 	mov	x2, #0x0                   	// #0
    3620:	17ffffe1 	b	35a4 <sptSparseTensorStatus+0x174>
    3624:	d503201f 	nop
    3628:	d503201f 	nop
    362c:	d503201f 	nop

0000000000003630 <sptMTTKRP_3D>:

int sptMTTKRP_3D(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode)
{
    3630:	a9b87bfd 	stp	x29, x30, [sp, #-128]!
    3634:	910003fd 	mov	x29, sp
    3638:	a9046bf9 	stp	x25, x26, [sp, #64]
    363c:	aa0003fa 	mov	x26, x0
    3640:	aa0103f9 	mov	x25, x1
	sptIndex const nmodes = X->nmodes;
	sptNnzIndex const nnz = X->nnz;
	sptIndex const * const ndims = X->ndims;
	sptValue const * const restrict vals = X->values.data;
	sptIndex const stride = mats[0]->stride;
    3644:	f9400020 	ldr	x0, [x1]
{
    3648:	a90153f3 	stp	x19, x20, [sp, #16]

	/* Check the mats. */
	sptAssert(nmodes ==3);
    364c:	b9400341 	ldr	w1, [x26]
{
    3650:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const stride = mats[0]->stride;
    3654:	b9400c14 	ldr	w20, [x0, #12]
{
    3658:	a90363f7 	stp	x23, x24, [sp, #48]
    365c:	a90573fb 	stp	x27, x28, [sp, #80]
    3660:	aa0203fc 	mov	x28, x2
	sptNnzIndex const nnz = X->nnz;
    3664:	a9415742 	ldp	x2, x21, [x26, #16]
	sptValue const * const restrict vals = X->values.data;
    3668:	f9401f57 	ldr	x23, [x26, #56]
	sptAssert(nmodes ==3);
    366c:	71000c3f 	cmp	w1, #0x3
    3670:	54001061 	b.ne	387c <sptMTTKRP_3D+0x24c>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3674:	f9400f24 	ldr	x4, [x25, #24]
    3678:	b9400405 	ldr	w5, [x0, #4]
    367c:	b9400481 	ldr	w1, [x4, #4]
    3680:	6b05003f 	cmp	w1, w5
    3684:	54000d01 	b.ne	3824 <sptMTTKRP_3D+0x1f4>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
		}
		if(mats[i]->nrows != ndims[i]) {
    3688:	b9400000 	ldr	w0, [x0]
    368c:	b9400045 	ldr	w5, [x2]
    3690:	6b0000bf 	cmp	w5, w0
    3694:	54000de1 	b.ne	3850 <sptMTTKRP_3D+0x220>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3698:	f9400720 	ldr	x0, [x25, #8]
    369c:	b9400405 	ldr	w5, [x0, #4]
    36a0:	6b05003f 	cmp	w1, w5
    36a4:	54000c01 	b.ne	3824 <sptMTTKRP_3D+0x1f4>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    36a8:	b9400000 	ldr	w0, [x0]
    36ac:	b9400445 	ldr	w5, [x2, #4]
    36b0:	6b0000bf 	cmp	w5, w0
    36b4:	54000ce1 	b.ne	3850 <sptMTTKRP_3D+0x220>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    36b8:	f9400b20 	ldr	x0, [x25, #16]
    36bc:	b9400405 	ldr	w5, [x0, #4]
    36c0:	6b0100bf 	cmp	w5, w1
    36c4:	54000b01 	b.ne	3824 <sptMTTKRP_3D+0x1f4>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    36c8:	b9400001 	ldr	w1, [x0]
    36cc:	b9400840 	ldr	w0, [x2, #8]
    36d0:	6b00003f 	cmp	w1, w0
    36d4:	54000be1 	b.ne	3850 <sptMTTKRP_3D+0x220>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}


	sptIndex const tmpI = mats[mode]->nrows;
    36d8:	f8635b20 	ldr	x0, [x25, w3, uxtw #3]
	sptIndex const R = mats[mode]->ncols;
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    36dc:	5280031b 	mov	w27, #0x18                  	// #24
    36e0:	f9401345 	ldr	x5, [x26, #32]
	sptMatrix * const restrict M = mats[nmodes];
	sptValue * const restrict mvals = M->values;
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    36e4:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    36e8:	29405802 	ldp	w2, w22, [x0]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    36ec:	9bbb1463 	umaddl	x3, w3, w27, x5
	sptValue * const restrict mvals = M->values;
    36f0:	f9400893 	ldr	x19, [x4, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    36f4:	1b027e82 	mul	w2, w20, w2
    36f8:	aa1303e0 	mov	x0, x19
    36fc:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3700:	f9400878 	ldr	x24, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3704:	fd0033e8 	str	d8, [sp, #96]
    3708:	97fff6ea 	bl	12b0 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
	sptIndex times_mat_index_2 = mats_order[2];
    370c:	29408f82 	ldp	w2, w3, [x28, #4]
	sptIndex mode_i;
	sptIndex tmp_i_1, tmp_i_2;
	sptValue entry;

	sptTimer timer;
	sptNewTimer(&timer, 0);
    3710:	9101e3e0 	add	x0, sp, #0x78
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3714:	f9401344 	ldr	x4, [x26, #32]
	sptNewTimer(&timer, 0);
    3718:	52800001 	mov	w1, #0x0                   	// #0
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    371c:	f8625b3c 	ldr	x28, [x25, w2, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3720:	9bbb1042 	umaddl	x2, w2, w27, x4
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3724:	9bbb107b 	umaddl	x27, w3, w27, x4
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    3728:	f8635b3a 	ldr	x26, [x25, w3, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    372c:	f9400859 	ldr	x25, [x2, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3730:	f9400b7b 	ldr	x27, [x27, #16]
	sptNewTimer(&timer, 0);
    3734:	94000363 	bl	44c0 <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    3738:	f9403fe0 	ldr	x0, [sp, #120]
    373c:	94000381 	bl	4540 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3740:	b4000435 	cbz	x21, 37c4 <sptMTTKRP_3D+0x194>
    3744:	34000416 	cbz	w22, 37c4 <sptMTTKRP_3D+0x194>
		tmp_i_1 = times_inds_1[x];
		tmp_i_2 = times_inds_2[x];
		entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3748:	f9400b47 	ldr	x7, [x26, #16]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    374c:	d2800009 	mov	x9, #0x0                   	// #0
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3750:	f9400b88 	ldr	x8, [x28, #16]
    3754:	d503201f 	nop
    3758:	b8697b04 	ldr	w4, [x24, x9, lsl #2]
    375c:	b8697b26 	ldr	w6, [x25, x9, lsl #2]
    3760:	b8697b65 	ldr	w5, [x27, x9, lsl #2]
		entry = vals[x];
    3764:	bc697ae3 	ldr	s3, [x23, x9, lsl #2]
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3768:	1b047e84 	mul	w4, w20, w4
    376c:	1b067e86 	mul	w6, w20, w6
    3770:	1b057e85 	mul	w5, w20, w5
    3774:	2a0403e0 	mov	w0, w4
    3778:	4b0000c6 	sub	w6, w6, w0
    377c:	0b160084 	add	w4, w4, w22
    3780:	4b0000a5 	sub	w5, w5, w0
    3784:	d503201f 	nop
    3788:	0b0000c3 	add	w3, w6, w0
    378c:	0b0000a2 	add	w2, w5, w0
    3790:	2a0003e1 	mov	w1, w0
		for(sptIndex r=0; r<R; ++r) {
    3794:	11000400 	add	w0, w0, #0x1
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3798:	bc637901 	ldr	s1, [x8, x3, lsl #2]
    379c:	bc6278e2 	ldr	s2, [x7, x2, lsl #2]
    37a0:	bc617a60 	ldr	s0, [x19, x1, lsl #2]
    37a4:	1e210861 	fmul	s1, s3, s1
    37a8:	1f020020 	fmadd	s0, s1, s2, s0
    37ac:	bc217a60 	str	s0, [x19, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    37b0:	6b04001f 	cmp	w0, w4
    37b4:	54fffea1 	b.ne	3788 <sptMTTKRP_3D+0x158>  // b.any
	for(sptNnzIndex x=0; x<nnz; ++x) {
    37b8:	91000529 	add	x9, x9, #0x1
    37bc:	eb0902bf 	cmp	x21, x9
    37c0:	54fffcc1 	b.ne	3758 <sptMTTKRP_3D+0x128>  // b.any
		}
	}
	sptStopTimer(timer);
    37c4:	f9403fe0 	ldr	x0, [sp, #120]
    37c8:	9400038e 	bl	4600 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    37cc:	f9403fe0 	ldr	x0, [sp, #120]
    37d0:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    37d4:	91148021 	add	x1, x1, #0x520
    37d8:	940003ca 	bl	4700 <sptPrintElapsedTime>
    37dc:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    37e0:	f9403fe0 	ldr	x0, [sp, #120]
    37e4:	94000403 	bl	47f0 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    37e8:	1e604100 	fmov	d0, d8
    37ec:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    37f0:	91156000 	add	x0, x0, #0x558
    37f4:	97fff703 	bl	1400 <printf@plt>
	printf("\n");
    37f8:	52800140 	mov	w0, #0xa                   	// #10
    37fc:	97fff70d 	bl	1430 <putchar@plt>

	return 0;
    3800:	fd4033e8 	ldr	d8, [sp, #96]
    3804:	52800000 	mov	w0, #0x0                   	// #0
}
    3808:	a94153f3 	ldp	x19, x20, [sp, #16]
    380c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3810:	a94363f7 	ldp	x23, x24, [sp, #48]
    3814:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3818:	a94573fb 	ldp	x27, x28, [sp, #80]
    381c:	a8c87bfd 	ldp	x29, x30, [sp], #128
    3820:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3824:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    3828:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    382c:	91134084 	add	x4, x4, #0x4d0
    3830:	9113e042 	add	x2, x2, #0x4f8
    3834:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    3838:	52801263 	mov	w3, #0x93                  	// #147
    383c:	91148000 	add	x0, x0, #0x520
    3840:	52800041 	mov	w1, #0x2                   	// #2
    3844:	9400046b 	bl	49f0 <spt_ComplainError>
    3848:	52800040 	mov	w0, #0x2                   	// #2
    384c:	17ffffef 	b	3808 <sptMTTKRP_3D+0x1d8>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3850:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    3854:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    3858:	9114e084 	add	x4, x4, #0x538
    385c:	9113e042 	add	x2, x2, #0x4f8
    3860:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    3864:	528012c3 	mov	w3, #0x96                  	// #150
    3868:	91148000 	add	x0, x0, #0x520
    386c:	52800041 	mov	w1, #0x2                   	// #2
    3870:	94000460 	bl	49f0 <spt_ComplainError>
    3874:	52800040 	mov	w0, #0x2                   	// #2
    3878:	17ffffe4 	b	3808 <sptMTTKRP_3D+0x1d8>
	sptAssert(nmodes ==3);
    387c:	12800000 	mov	w0, #0xffffffff            	// #-1
    3880:	fd0033e8 	str	d8, [sp, #96]
    3884:	97fff64f 	bl	11c0 <exit@plt>
    3888:	d503201f 	nop
    388c:	d503201f 	nop

0000000000003890 <sptMTTKRP>:
							sptIndex const mode) {
    3890:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    3894:	910003fd 	mov	x29, sp
    3898:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const nmodes = X->nmodes;
    389c:	b9400015 	ldr	w21, [x0]
	if(nmodes == 3) {
    38a0:	71000ebf 	cmp	w21, #0x3
    38a4:	54002060 	b.eq	3cb0 <sptMTTKRP+0x420>  // b.none
	sptNnzIndex const nnz = X->nnz;
    38a8:	a90363f7 	stp	x23, x24, [sp, #48]
    38ac:	aa0103f6 	mov	x22, x1
    38b0:	aa0203f8 	mov	x24, x2
    38b4:	a90573fb 	stp	x27, x28, [sp, #80]
    38b8:	2a0303e5 	mov	w5, w3
    38bc:	aa0003fc 	mov	x28, x0
	sptMatrix * const restrict M = mats[nmodes];
    38c0:	f8755826 	ldr	x6, [x1, w21, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    38c4:	f9400804 	ldr	x4, [x0, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    38c8:	340003d5 	cbz	w21, 3940 <sptMTTKRP+0xb0>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    38cc:	b94004c3 	ldr	w3, [x6, #4]
    38d0:	d2800000 	mov	x0, #0x0                   	// #0
    38d4:	14000008 	b	38f4 <sptMTTKRP+0x64>
		if(mats[i]->nrows != ndims[i]) {
    38d8:	b9400022 	ldr	w2, [x1]
    38dc:	b8607881 	ldr	w1, [x4, x0, lsl #2]
    38e0:	6b01005f 	cmp	w2, w1
    38e4:	54001d81 	b.ne	3c94 <sptMTTKRP+0x404>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    38e8:	91000400 	add	x0, x0, #0x1
    38ec:	6b0002bf 	cmp	w21, w0
    38f0:	54000289 	b.ls	3940 <sptMTTKRP+0xb0>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    38f4:	f8607ac1 	ldr	x1, [x22, x0, lsl #3]
    38f8:	b9400422 	ldr	w2, [x1, #4]
    38fc:	6b03005f 	cmp	w2, w3
    3900:	54fffec0 	b.eq	38d8 <sptMTTKRP+0x48>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3904:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    3908:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    390c:	91134084 	add	x4, x4, #0x4d0
    3910:	9113e042 	add	x2, x2, #0x4f8
    3914:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    3918:	528007c3 	mov	w3, #0x3e                  	// #62
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    391c:	91148000 	add	x0, x0, #0x520
    3920:	52800041 	mov	w1, #0x2                   	// #2
    3924:	94000433 	bl	49f0 <spt_ComplainError>
    3928:	a94363f7 	ldp	x23, x24, [sp, #48]
    392c:	52800040 	mov	w0, #0x2                   	// #2
    3930:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3934:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3938:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    393c:	d65f03c0 	ret
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3940:	f9401383 	ldr	x3, [x28, #32]
    3944:	a9046bf9 	stp	x25, x26, [sp, #64]
    3948:	52800319 	mov	w25, #0x18                  	// #24
	sptIndex const tmpI = mats[mode]->nrows;
    394c:	f8655ac0 	ldr	x0, [x22, w5, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3950:	9bb90ca3 	umaddl	x3, w5, w25, x3
	sptIndex const stride = mats[0]->stride;
    3954:	f94002c4 	ldr	x4, [x22]
	sptNewValueVector(&scratch, R, R);
    3958:	910223e2 	add	x2, sp, #0x88
    395c:	a90153f3 	stp	x19, x20, [sp, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3960:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    3964:	b9400413 	ldr	w19, [x0, #4]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3968:	f940087b 	ldr	x27, [x3, #16]
	sptNnzIndex const nnz = X->nnz;
    396c:	f9400f83 	ldr	x3, [x28, #24]
	sptNewValueVector(&scratch, R, R);
    3970:	a9070be3 	stp	x3, x2, [sp, #112]
	sptIndex const stride = mats[0]->stride;
    3974:	b9400c97 	ldr	w23, [x4, #12]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3978:	b9400002 	ldr	w2, [x0]
	sptValue * const restrict mvals = M->values;
    397c:	f94008d4 	ldr	x20, [x6, #16]
	sptValue const * const restrict vals = X->values.data;
    3980:	f9401f9a 	ldr	x26, [x28, #56]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3984:	1b027ee2 	mul	w2, w23, w2
    3988:	aa1403e0 	mov	x0, x20
    398c:	fd0033e8 	str	d8, [sp, #96]
    3990:	d37ef442 	lsl	x2, x2, #2
    3994:	97fff647 	bl	12b0 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    3998:	f9403fe0 	ldr	x0, [sp, #120]
    399c:	2a1303e2 	mov	w2, w19
    39a0:	aa0203e1 	mov	x1, x2
    39a4:	97fff9a3 	bl	2030 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    39a8:	0f000400 	movi	v0.2s, #0x0
    39ac:	f9403fe0 	ldr	x0, [sp, #120]
    39b0:	97fff9c8 	bl	20d0 <sptConstantValueVector>
	sptNewTimer(&timer, 0);
    39b4:	52800001 	mov	w1, #0x0                   	// #0
    39b8:	910203e0 	add	x0, sp, #0x80
    39bc:	940002c1 	bl	44c0 <sptNewTimer>
	sptStartTimer(timer);
    39c0:	f94043e0 	ldr	x0, [sp, #128]
    39c4:	940002df 	bl	4540 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    39c8:	f9403be3 	ldr	x3, [sp, #112]
    39cc:	b40010a3 	cbz	x3, 3be0 <sptMTTKRP+0x350>
		sptIndex times_mat_index = mats_order[1];
    39d0:	b9400700 	ldr	w0, [x24, #4]
    39d4:	53027e64 	lsr	w4, w19, #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    39d8:	f940138b 	ldr	x11, [x28, #32]
    39dc:	121e766d 	and	w13, w19, #0xfffffffc
			scratch.data[r] = entry * times_mat_values[r];
    39e0:	f9404fe1 	ldr	x1, [sp, #152]
    39e4:	51000484 	sub	w4, w4, #0x1
		sptIndex * times_inds = X->inds[times_mat_index].data;
    39e8:	9bb92c02 	umaddl	x2, w0, w25, x11
		for(sptIndex r=0; r<R; ++r) {
    39ec:	110009b2 	add	w18, w13, #0x2
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    39f0:	f8605ac0 	ldr	x0, [x22, w0, uxtw #3]
    39f4:	91000484 	add	x4, x4, #0x1
			scratch.data[r] = entry * times_mat_values[r];
    39f8:	110005b0 	add	w16, w13, #0x1
    39fc:	927e766e 	and	x14, x19, #0xfffffffc
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3a00:	f940085c 	ldr	x28, [x2, #16]
		for(sptIndex r=0; r<R; ++r) {
    3a04:	aa1203f1 	mov	x17, x18
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3a08:	f940081e 	ldr	x30, [x0, #16]
    3a0c:	d37cec84 	lsl	x4, x4, #4
		for(sptIndex r=0; r<R; ++r) {
    3a10:	110005af 	add	w15, w13, #0x1
    3a14:	5100066a 	sub	w10, w19, #0x1
			scratch.data[r] = entry * times_mat_values[r];
    3a18:	8b12082c 	add	x12, x1, x18, lsl #2
    3a1c:	8b100829 	add	x9, x1, x16, lsl #2
    3a20:	8b0e0828 	add	x8, x1, x14, lsl #2
    3a24:	d37ef463 	lsl	x3, x3, #2
    3a28:	d2800007 	mov	x7, #0x0                   	// #0
    3a2c:	f9003be3 	str	x3, [sp, #112]
		sptIndex tmp_mult = tmp_i * stride;
    3a30:	b8676b83 	ldr	w3, [x28, x7]
		sptValue const entry = vals[x];
    3a34:	bc676b42 	ldr	s2, [x26, x7]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3a38:	1b037ee3 	mul	w3, w23, w3
		for(sptIndex r=0; r<R; ++r) {
    3a3c:	34000cb3 	cbz	w19, 3bd0 <sptMTTKRP+0x340>
    3a40:	91000460 	add	x0, x3, #0x1
    3a44:	d37ef400 	lsl	x0, x0, #2
    3a48:	8b0003c2 	add	x2, x30, x0
    3a4c:	cb020022 	sub	x2, x1, x2
    3a50:	f100205f 	cmp	x2, #0x8
    3a54:	7a458940 	ccmp	w10, #0x5, #0x0, hi  // hi = pmore
    3a58:	540010c9 	b.ls	3c70 <sptMTTKRP+0x3e0>  // b.plast
    3a5c:	4e040441 	dup	v1.4s, v2.s[0]
    3a60:	d1001002 	sub	x2, x0, #0x4
    3a64:	8b0203c2 	add	x2, x30, x2
    3a68:	d2800000 	mov	x0, #0x0                   	// #0
    3a6c:	d503201f 	nop
			scratch.data[r] = entry * times_mat_values[r];
    3a70:	3ce06840 	ldr	q0, [x2, x0]
    3a74:	6e21dc00 	fmul	v0.4s, v0.4s, v1.4s
    3a78:	3ca06820 	str	q0, [x1, x0]
		for(sptIndex r=0; r<R; ++r) {
    3a7c:	91004000 	add	x0, x0, #0x10
    3a80:	eb04001f 	cmp	x0, x4
    3a84:	54ffff61 	b.ne	3a70 <sptMTTKRP+0x1e0>  // b.any
    3a88:	6b1301bf 	cmp	w13, w19
    3a8c:	54000220 	b.eq	3ad0 <sptMTTKRP+0x240>  // b.none
			scratch.data[r] = entry * times_mat_values[r];
    3a90:	8b0e0060 	add	x0, x3, x14
    3a94:	bc607bc0 	ldr	s0, [x30, x0, lsl #2]
    3a98:	1e220800 	fmul	s0, s0, s2
    3a9c:	bd000100 	str	s0, [x8]
		for(sptIndex r=0; r<R; ++r) {
    3aa0:	6b0f027f 	cmp	w19, w15
    3aa4:	54000169 	b.ls	3ad0 <sptMTTKRP+0x240>  // b.plast
			scratch.data[r] = entry * times_mat_values[r];
    3aa8:	8b100060 	add	x0, x3, x16
    3aac:	bc607bc0 	ldr	s0, [x30, x0, lsl #2]
    3ab0:	1e220800 	fmul	s0, s0, s2
    3ab4:	bd000120 	str	s0, [x9]
		for(sptIndex r=0; r<R; ++r) {
    3ab8:	6b11027f 	cmp	w19, w17
    3abc:	540000a9 	b.ls	3ad0 <sptMTTKRP+0x240>  // b.plast
			scratch.data[r] = entry * times_mat_values[r];
    3ac0:	8b120063 	add	x3, x3, x18
    3ac4:	bc637bc0 	ldr	s0, [x30, x3, lsl #2]
    3ac8:	1e220802 	fmul	s2, s0, s2
    3acc:	bd000182 	str	s2, [x12]
		for(sptIndex i=2; i<nmodes; ++i) {
    3ad0:	d2800046 	mov	x6, #0x2                   	// #2
    3ad4:	71000abf 	cmp	w21, #0x2
    3ad8:	54000669 	b.ls	3ba4 <sptMTTKRP+0x314>  // b.plast
    3adc:	d503201f 	nop
			times_mat_index = mats_order[i];
    3ae0:	b8667b00 	ldr	w0, [x24, x6, lsl #2]
			tmp_i = times_inds[x];
    3ae4:	9bb92c03 	umaddl	x3, w0, w25, x11
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3ae8:	f8605ac2 	ldr	x2, [x22, w0, uxtw #3]
			tmp_i = times_inds[x];
    3aec:	f9400860 	ldr	x0, [x3, #16]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3af0:	f9400845 	ldr	x5, [x2, #16]
			tmp_mult = tmp_i * stride;
    3af4:	b8676803 	ldr	w3, [x0, x7]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3af8:	1b037ee3 	mul	w3, w23, w3
			for(sptIndex r=0; r<R; ++r) {
    3afc:	91000462 	add	x2, x3, #0x1
    3b00:	d37ef442 	lsl	x2, x2, #2
    3b04:	8b0200a0 	add	x0, x5, x2
    3b08:	cb000020 	sub	x0, x1, x0
    3b0c:	f100201f 	cmp	x0, #0x8
    3b10:	7a448940 	ccmp	w10, #0x4, #0x0, hi  // hi = pmore
    3b14:	540009a9 	b.ls	3c48 <sptMTTKRP+0x3b8>  // b.plast
    3b18:	d1001042 	sub	x2, x2, #0x4
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3b1c:	d2800000 	mov	x0, #0x0                   	// #0
    3b20:	8b0200a2 	add	x2, x5, x2
    3b24:	d503201f 	nop
				scratch_data[r] *= times_mat_values[r];
    3b28:	3ce06840 	ldr	q0, [x2, x0]
    3b2c:	3ce06821 	ldr	q1, [x1, x0]
    3b30:	6e21dc00 	fmul	v0.4s, v0.4s, v1.4s
    3b34:	3ca06820 	str	q0, [x1, x0]
			for(sptIndex r=0; r<R; ++r) {
    3b38:	91004000 	add	x0, x0, #0x10
    3b3c:	eb04001f 	cmp	x0, x4
    3b40:	54ffff41 	b.ne	3b28 <sptMTTKRP+0x298>  // b.any
    3b44:	6b1301bf 	cmp	w13, w19
    3b48:	54000280 	b.eq	3b98 <sptMTTKRP+0x308>  // b.none
				scratch_data[r] *= times_mat_values[r];
    3b4c:	8b0e0060 	add	x0, x3, x14
    3b50:	bd400100 	ldr	s0, [x8]
    3b54:	bc6078a1 	ldr	s1, [x5, x0, lsl #2]
    3b58:	1e210800 	fmul	s0, s0, s1
    3b5c:	bd000100 	str	s0, [x8]
			for(sptIndex r=0; r<R; ++r) {
    3b60:	6b0f027f 	cmp	w19, w15
    3b64:	540001a9 	b.ls	3b98 <sptMTTKRP+0x308>  // b.plast
				scratch_data[r] *= times_mat_values[r];
    3b68:	8b100060 	add	x0, x3, x16
    3b6c:	bd400120 	ldr	s0, [x9]
    3b70:	bc6078a1 	ldr	s1, [x5, x0, lsl #2]
    3b74:	1e210800 	fmul	s0, s0, s1
    3b78:	bd000120 	str	s0, [x9]
			for(sptIndex r=0; r<R; ++r) {
    3b7c:	6b11027f 	cmp	w19, w17
    3b80:	540000c9 	b.ls	3b98 <sptMTTKRP+0x308>  // b.plast
				scratch_data[r] *= times_mat_values[r];
    3b84:	8b120063 	add	x3, x3, x18
    3b88:	bd400180 	ldr	s0, [x12]
    3b8c:	bc6378a1 	ldr	s1, [x5, x3, lsl #2]
    3b90:	1e210800 	fmul	s0, s0, s1
    3b94:	bd000180 	str	s0, [x12]
		for(sptIndex i=2; i<nmodes; ++i) {
    3b98:	910004c6 	add	x6, x6, #0x1
    3b9c:	6b0602bf 	cmp	w21, w6
    3ba0:	54fffa08 	b.hi	3ae0 <sptMTTKRP+0x250>  // b.pmore
		sptIndex tmp_mode = mode_i * stride;
    3ba4:	b8676b63 	ldr	w3, [x27, x7]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3ba8:	d2800000 	mov	x0, #0x0                   	// #0
		sptIndex tmp_mode = mode_i * stride;
    3bac:	1b037ee3 	mul	w3, w23, w3
			mvals[tmp_mode + r] += scratch_data[r];
    3bb0:	0b000062 	add	w2, w3, w0
    3bb4:	bc607821 	ldr	s1, [x1, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3bb8:	91000400 	add	x0, x0, #0x1
			mvals[tmp_mode + r] += scratch_data[r];
    3bbc:	bc627a80 	ldr	s0, [x20, x2, lsl #2]
    3bc0:	1e212800 	fadd	s0, s0, s1
    3bc4:	bc227a80 	str	s0, [x20, x2, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3bc8:	6b00027f 	cmp	w19, w0
    3bcc:	54ffff28 	b.hi	3bb0 <sptMTTKRP+0x320>  // b.pmore
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3bd0:	f9403be0 	ldr	x0, [sp, #112]
    3bd4:	910010e7 	add	x7, x7, #0x4
    3bd8:	eb0000ff 	cmp	x7, x0
    3bdc:	54fff2a1 	b.ne	3a30 <sptMTTKRP+0x1a0>  // b.any
	sptStopTimer(timer);
    3be0:	f94043e0 	ldr	x0, [sp, #128]
    3be4:	94000287 	bl	4600 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    3be8:	f94043e0 	ldr	x0, [sp, #128]
    3bec:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    3bf0:	91148021 	add	x1, x1, #0x520
    3bf4:	940002c3 	bl	4700 <sptPrintElapsedTime>
    3bf8:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3bfc:	f94043e0 	ldr	x0, [sp, #128]
    3c00:	940002fc 	bl	47f0 <sptFreeTimer>
	sptFreeValueVector(&scratch);
    3c04:	f9403fe0 	ldr	x0, [sp, #120]
    3c08:	97fff9b7 	bl	22e4 <sptFreeValueVector>
	printf("[Total time]: %lf\n", total_time);
    3c0c:	1e604100 	fmov	d0, d8
    3c10:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    3c14:	91156000 	add	x0, x0, #0x558
    3c18:	97fff5fa 	bl	1400 <printf@plt>
	printf("\n");
    3c1c:	52800140 	mov	w0, #0xa                   	// #10
    3c20:	97fff604 	bl	1430 <putchar@plt>
	return 0;
    3c24:	52800000 	mov	w0, #0x0                   	// #0
    3c28:	fd4033e8 	ldr	d8, [sp, #96]
    3c2c:	a94153f3 	ldp	x19, x20, [sp, #16]
}
    3c30:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    3c34:	a94363f7 	ldp	x23, x24, [sp, #48]
    3c38:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3c3c:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3c40:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3c44:	d65f03c0 	ret
    3c48:	8b0308a3 	add	x3, x5, x3, lsl #2
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    3c4c:	d2800000 	mov	x0, #0x0                   	// #0
				scratch_data[r] *= times_mat_values[r];
    3c50:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    3c54:	bc607861 	ldr	s1, [x3, x0, lsl #2]
    3c58:	1e210800 	fmul	s0, s0, s1
    3c5c:	bc207820 	str	s0, [x1, x0, lsl #2]
			for(sptIndex r=0; r<R; ++r) {
    3c60:	91000400 	add	x0, x0, #0x1
    3c64:	6b00027f 	cmp	w19, w0
    3c68:	54ffff48 	b.hi	3c50 <sptMTTKRP+0x3c0>  // b.pmore
    3c6c:	17ffffcb 	b	3b98 <sptMTTKRP+0x308>
    3c70:	8b030bc3 	add	x3, x30, x3, lsl #2
		for(sptIndex r=0; r<R; ++r) {
    3c74:	d2800000 	mov	x0, #0x0                   	// #0
			scratch.data[r] = entry * times_mat_values[r];
    3c78:	bc607860 	ldr	s0, [x3, x0, lsl #2]
    3c7c:	1e220800 	fmul	s0, s0, s2
    3c80:	bc207820 	str	s0, [x1, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3c84:	91000400 	add	x0, x0, #0x1
    3c88:	6b00027f 	cmp	w19, w0
    3c8c:	54ffff68 	b.hi	3c78 <sptMTTKRP+0x3e8>  // b.pmore
    3c90:	17ffff90 	b	3ad0 <sptMTTKRP+0x240>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3c94:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    3c98:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    3c9c:	9114e084 	add	x4, x4, #0x538
    3ca0:	9113e042 	add	x2, x2, #0x4f8
    3ca4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    3ca8:	52800823 	mov	w3, #0x41                  	// #65
    3cac:	17ffff1c 	b	391c <sptMTTKRP+0x8c>
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    3cb0:	97fffe60 	bl	3630 <sptMTTKRP_3D>
    3cb4:	34ffe400 	cbz	w0, 3934 <sptMTTKRP+0xa4>
    3cb8:	12800000 	mov	w0, #0xffffffff            	// #-1
    3cbc:	a90153f3 	stp	x19, x20, [sp, #16]
    3cc0:	a90363f7 	stp	x23, x24, [sp, #48]
    3cc4:	a9046bf9 	stp	x25, x26, [sp, #64]
    3cc8:	a90573fb 	stp	x27, x28, [sp, #80]
    3ccc:	fd0033e8 	str	d8, [sp, #96]
    3cd0:	97fff53c 	bl	11c0 <exit@plt>
    3cd4:	d503201f 	nop
    3cd8:	d503201f 	nop
    3cdc:	d503201f 	nop

0000000000003ce0 <sptOmpMTTKRP._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3ce0:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    3ce4:	910003fd 	mov	x29, sp
    3ce8:	a9025bf5 	stp	x21, x22, [sp, #32]
    3cec:	f9400c15 	ldr	x21, [x0, #24]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3cf0:	b4001075 	cbz	x21, 3efc <sptOmpMTTKRP._omp_fn.0+0x21c>
    3cf4:	a90153f3 	stp	x19, x20, [sp, #16]
    3cf8:	aa0003f4 	mov	x20, x0
    3cfc:	97fff581 	bl	1300 <omp_get_num_threads@plt>
    3d00:	93407c13 	sxtw	x19, w0
    3d04:	97fff547 	bl	1220 <omp_get_thread_num@plt>
    3d08:	93407c16 	sxtw	x22, w0
    3d0c:	9ad30aa1 	udiv	x1, x21, x19
    3d10:	9b13d420 	msub	x0, x1, x19, x21
    3d14:	eb0002df 	cmp	x22, x0
    3d18:	54001003 	b.cc	3f18 <sptOmpMTTKRP._omp_fn.0+0x238>  // b.lo, b.ul, b.last
    3d1c:	9b160036 	madd	x22, x1, x22, x0
    3d20:	8b160020 	add	x0, x1, x22
    3d24:	f90043e0 	str	x0, [sp, #128]
    3d28:	eb0002df 	cmp	x22, x0
    3d2c:	54000ee2 	b.cs	3f08 <sptOmpMTTKRP._omp_fn.0+0x228>  // b.hs, b.nlast
    3d30:	9102e3e0 	add	x0, sp, #0xb8
    3d34:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    3d38:	b940429c 	ldr	w28, [x20, #64]
    3d3c:	f9003fe0 	str	x0, [sp, #120]
    3d40:	f9400280 	ldr	x0, [x20]
    3d44:	f9004be0 	str	x0, [sp, #144]
    3d48:	b9403a80 	ldr	w0, [x20, #56]
    3d4c:	51000795 	sub	w21, w28, #0x1
    3d50:	b90077e0 	str	w0, [sp, #116]
		sptValueVector scratch;  // Temporary array
		sptNewValueVector(&scratch, R, R);
    3d54:	2a1c03e0 	mov	w0, w28
    3d58:	f9004fe0 	str	x0, [sp, #152]
    3d5c:	910006b5 	add	x21, x21, #0x1
#pragma omp parallel for schedule(static) num_threads(tk)
    3d60:	f9401280 	ldr	x0, [x20, #32]
    3d64:	f90047e0 	str	x0, [sp, #136]
    3d68:	f9401680 	ldr	x0, [x20, #40]
    3d6c:	a90363f7 	stp	x23, x24, [sp, #48]
    3d70:	d37ef6b5 	lsl	x21, x21, #2
    3d74:	a9046bf9 	stp	x25, x26, [sp, #64]
    3d78:	f90057e0 	str	x0, [sp, #168]
    3d7c:	a940e299 	ldp	x25, x24, [x20, #8]
    3d80:	fd0033e8 	str	d8, [sp, #96]
    3d84:	f9401a80 	ldr	x0, [x20, #48]
    3d88:	f90053e0 	str	x0, [sp, #160]
    3d8c:	b9403e94 	ldr	w20, [x20, #60]
		sptNewValueVector(&scratch, R, R);
    3d90:	f9404fe1 	ldr	x1, [sp, #152]
    3d94:	f9403ff3 	ldr	x19, [sp, #120]
    3d98:	aa0103e2 	mov	x2, x1
    3d9c:	aa1303e0 	mov	x0, x19
    3da0:	97fff8a4 	bl	2030 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    3da4:	0f000400 	movi	v0.2s, #0x0
    3da8:	aa1303e0 	mov	x0, x19
    3dac:	97fff8c9 	bl	20d0 <sptConstantValueVector>

		sptIndex times_mat_index = mats_order[1];
		sptMatrix * times_mat = mats[times_mat_index];
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3db0:	f9404be1 	ldr	x1, [sp, #144]
    3db4:	d37ef6c8 	lsl	x8, x22, #2
		sptIndex times_mat_index = mats_order[1];
    3db8:	b9400700 	ldr	w0, [x24, #4]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3dbc:	f9401027 	ldr	x7, [x1, #32]
		sptIndex tmp_i = times_inds[x];
		sptValue const entry = vals[x];
    3dc0:	f94047e1 	ldr	x1, [sp, #136]
		sptMatrix * times_mat = mats[times_mat_index];
    3dc4:	f8605b22 	ldr	x2, [x25, w0, uxtw #3]
		sptValue const entry = vals[x];
    3dc8:	bc767821 	ldr	s1, [x1, x22, lsl #2]
		sptIndex tmp_i = times_inds[x];
    3dcc:	52800301 	mov	w1, #0x18                  	// #24
    3dd0:	9ba11c00 	umaddl	x0, w0, w1, x7
    3dd4:	f9400800 	ldr	x0, [x0, #16]
    3dd8:	b8767803 	ldr	w3, [x0, x22, lsl #2]
#pragma omp simd
		for(sptIndex r=0; r<R; ++r) {
    3ddc:	340007bc 	cbz	w28, 3ed0 <sptOmpMTTKRP._omp_fn.0+0x1f0>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    3de0:	1b037e83 	mul	w3, w20, w3
    3de4:	d2800000 	mov	x0, #0x0                   	// #0
    3de8:	f9400845 	ldr	x5, [x2, #16]
    3dec:	f94067e1 	ldr	x1, [sp, #200]
    3df0:	0b000062 	add	w2, w3, w0
    3df4:	bc6278a0 	ldr	s0, [x5, x2, lsl #2]
    3df8:	1e210800 	fmul	s0, s0, s1
    3dfc:	bc207820 	str	s0, [x1, x0, lsl #2]
    3e00:	91000400 	add	x0, x0, #0x1
    3e04:	6b00039f 	cmp	w28, w0
    3e08:	54ffff48 	b.hi	3df0 <sptOmpMTTKRP._omp_fn.0+0x110>  // b.pmore
		}

		for(sptIndex i=2; i<nmodes; ++i) {
    3e0c:	b94077e0 	ldr	w0, [sp, #116]
			times_inds = X->inds[times_mat_index].data;
			tmp_i = times_inds[x];

#pragma omp simd
			for(sptIndex r=0; r<R; ++r) {
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3e10:	d2800046 	mov	x6, #0x2                   	// #2
		for(sptIndex i=2; i<nmodes; ++i) {
    3e14:	7100081f 	cmp	w0, #0x2
    3e18:	54000309 	b.ls	3e78 <sptOmpMTTKRP._omp_fn.0+0x198>  // b.plast
    3e1c:	d503201f 	nop
			times_mat_index = mats_order[i];
    3e20:	b8667b02 	ldr	w2, [x24, x6, lsl #2]
			tmp_i = times_inds[x];
    3e24:	52800303 	mov	w3, #0x18                  	// #24
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3e28:	d2800000 	mov	x0, #0x0                   	// #0
			tmp_i = times_inds[x];
    3e2c:	9ba31c45 	umaddl	x5, w2, w3, x7
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3e30:	f8625b23 	ldr	x3, [x25, w2, uxtw #3]
			tmp_i = times_inds[x];
    3e34:	f94008a2 	ldr	x2, [x5, #16]
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3e38:	f9400865 	ldr	x5, [x3, #16]
    3e3c:	b8686843 	ldr	w3, [x2, x8]
    3e40:	1b037e83 	mul	w3, w20, w3
    3e44:	d503201f 	nop
    3e48:	0b000062 	add	w2, w3, w0
    3e4c:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    3e50:	bc6278a1 	ldr	s1, [x5, x2, lsl #2]
    3e54:	1e210800 	fmul	s0, s0, s1
    3e58:	bc207820 	str	s0, [x1, x0, lsl #2]
    3e5c:	91000400 	add	x0, x0, #0x1
    3e60:	6b00039f 	cmp	w28, w0
    3e64:	54ffff28 	b.hi	3e48 <sptOmpMTTKRP._omp_fn.0+0x168>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    3e68:	b94077e0 	ldr	w0, [sp, #116]
    3e6c:	910004c6 	add	x6, x6, #0x1
    3e70:	6b06001f 	cmp	w0, w6
    3e74:	54fffd68 	b.hi	3e20 <sptOmpMTTKRP._omp_fn.0+0x140>  // b.pmore
			}
		}

		sptIndex const mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3e78:	f94057e0 	ldr	x0, [sp, #168]
#pragma omp parallel for schedule(static) num_threads(tk)
    3e7c:	d280001b 	mov	x27, #0x0                   	// #0
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3e80:	b8767813 	ldr	w19, [x0, x22, lsl #2]
    3e84:	f94053e0 	ldr	x0, [sp, #160]
    3e88:	1b137e93 	mul	w19, w20, w19
		for(sptIndex r=0; r<R; ++r) {
    3e8c:	8b130813 	add	x19, x0, x19, lsl #2
    3e90:	14000002 	b	3e98 <sptOmpMTTKRP._omp_fn.0+0x1b8>
    3e94:	f94067e1 	ldr	x1, [sp, #200]
    3e98:	8b130377 	add	x23, x27, x19
#pragma omp atomic update
			mvals_row[r] += scratch.data[r];
    3e9c:	bc7b6828 	ldr	s8, [x1, x27]
    3ea0:	b94002fa 	ldr	w26, [x23]
    3ea4:	1e270340 	fmov	s0, w26
    3ea8:	aa1703e2 	mov	x2, x23
    3eac:	2a1a03e0 	mov	w0, w26
    3eb0:	1e202900 	fadd	s0, s8, s0
    3eb4:	1e260001 	fmov	w1, s0
    3eb8:	9400035e 	bl	4c30 <__aarch64_cas4_acq_rel>
    3ebc:	6b00035f 	cmp	w26, w0
    3ec0:	54000321 	b.ne	3f24 <sptOmpMTTKRP._omp_fn.0+0x244>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    3ec4:	9100137b 	add	x27, x27, #0x4
    3ec8:	eb15037f 	cmp	x27, x21
    3ecc:	54fffe41 	b.ne	3e94 <sptOmpMTTKRP._omp_fn.0+0x1b4>  // b.any
		}

		sptFreeValueVector(&scratch);
    3ed0:	f9403fe0 	ldr	x0, [sp, #120]
    3ed4:	910006d6 	add	x22, x22, #0x1
    3ed8:	97fff903 	bl	22e4 <sptFreeValueVector>
    3edc:	f94043e0 	ldr	x0, [sp, #128]
    3ee0:	eb16001f 	cmp	x0, x22
    3ee4:	54fff568 	b.hi	3d90 <sptOmpMTTKRP._omp_fn.0+0xb0>  // b.pmore
    3ee8:	a94153f3 	ldp	x19, x20, [sp, #16]
    3eec:	a94363f7 	ldp	x23, x24, [sp, #48]
    3ef0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3ef4:	a94573fb 	ldp	x27, x28, [sp, #80]
    3ef8:	fd4033e8 	ldr	d8, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3efc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3f00:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3f04:	d65f03c0 	ret
    3f08:	a94153f3 	ldp	x19, x20, [sp, #16]
    3f0c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3f10:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3f14:	d65f03c0 	ret
    3f18:	91000421 	add	x1, x1, #0x1
    3f1c:	d2800000 	mov	x0, #0x0                   	// #0
    3f20:	17ffff7f 	b	3d1c <sptOmpMTTKRP._omp_fn.0+0x3c>
			mvals_row[r] += scratch.data[r];
    3f24:	2a0003fa 	mov	w26, w0
    3f28:	17ffffdf 	b	3ea4 <sptOmpMTTKRP._omp_fn.0+0x1c4>
    3f2c:	d503201f 	nop

0000000000003f30 <sptOmpMTTKRP_3D._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3f30:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    3f34:	910003fd 	mov	x29, sp
    3f38:	a9025bf5 	stp	x21, x22, [sp, #32]
    3f3c:	f9400016 	ldr	x22, [x0]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3f40:	b40009f6 	cbz	x22, 407c <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
    3f44:	a90153f3 	stp	x19, x20, [sp, #16]
    3f48:	aa0003f3 	mov	x19, x0
    3f4c:	a90363f7 	stp	x23, x24, [sp, #48]
    3f50:	97fff4ec 	bl	1300 <omp_get_num_threads@plt>
    3f54:	93407c14 	sxtw	x20, w0
    3f58:	97fff4b2 	bl	1220 <omp_get_thread_num@plt>
    3f5c:	93407c15 	sxtw	x21, w0
    3f60:	9ad40ad7 	udiv	x23, x22, x20
    3f64:	9b14dae0 	msub	x0, x23, x20, x22
    3f68:	eb0002bf 	cmp	x21, x0
    3f6c:	54000983 	b.cc	409c <sptOmpMTTKRP_3D._omp_fn.0+0x16c>  // b.lo, b.ul, b.last
    3f70:	9b1502f5 	madd	x21, x23, x21, x0
    3f74:	8b1502e0 	add	x0, x23, x21
    3f78:	f9003be0 	str	x0, [sp, #112]
    3f7c:	eb0002bf 	cmp	x21, x0
    3f80:	54000842 	b.cs	4088 <sptOmpMTTKRP_3D._omp_fn.0+0x158>  // b.hs, b.nlast
    3f84:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    3f88:	a940ee61 	ldp	x1, x27, [x19, #8]
    3f8c:	f9004fe1 	str	x1, [sp, #152]
    3f90:	f9400e61 	ldr	x1, [x19, #24]
    3f94:	f90047e1 	str	x1, [sp, #136]
    3f98:	29480276 	ldp	w22, w0, [x19, #64]
    3f9c:	b90097e0 	str	w0, [sp, #148]
    3fa0:	f9401661 	ldr	x1, [x19, #40]
    3fa4:	f90043e1 	str	x1, [sp, #128]
    3fa8:	f9401e61 	ldr	x1, [x19, #56]
    3fac:	f9003fe1 	str	x1, [sp, #120]
    3fb0:	f9401278 	ldr	x24, [x19, #32]
    3fb4:	f9401a7c 	ldr	x28, [x19, #48]
    3fb8:	34000780 	cbz	w0, 40a8 <sptOmpMTTKRP_3D._omp_fn.0+0x178>
    3fbc:	a9046bf9 	stp	x25, x26, [sp, #64]
    3fc0:	6d0627e8 	stp	d8, d9, [sp, #96]
    3fc4:	d503201f 	nop
		sptIndex mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3fc8:	b8757b62 	ldr	w2, [x27, x21, lsl #2]
		sptIndex tmp_i_2 = times_inds_2[x];
		sptValue entry = vals[x];

		for(sptIndex r=0; r<R; ++r) {
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3fcc:	f9403fe0 	ldr	x0, [sp, #120]
    3fd0:	1b027ec2 	mul	w2, w22, w2
    3fd4:	b8757813 	ldr	w19, [x0, x21, lsl #2]
    3fd8:	f94043e0 	ldr	x0, [sp, #128]
    3fdc:	1b137ed3 	mul	w19, w22, w19
    3fe0:	b8757814 	ldr	w20, [x0, x21, lsl #2]
		sptValue entry = vals[x];
    3fe4:	f9404fe0 	ldr	x0, [sp, #152]
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3fe8:	2a1303f9 	mov	w25, w19
    3fec:	1b147ed4 	mul	w20, w22, w20
		sptValue entry = vals[x];
    3ff0:	bc757809 	ldr	s9, [x0, x21, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3ff4:	b94097e0 	ldr	w0, [sp, #148]
    3ff8:	4b130294 	sub	w20, w20, w19
    3ffc:	0b130013 	add	w19, w0, w19
    4000:	f94047e0 	ldr	x0, [sp, #136]
    4004:	8b020817 	add	x23, x0, x2, lsl #2
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4008:	f9400b04 	ldr	x4, [x24, #16]
    400c:	0b190280 	add	w0, w20, w25
    4010:	f9400b81 	ldr	x1, [x28, #16]
    4014:	bc607888 	ldr	s8, [x4, x0, lsl #2]
    4018:	bc795820 	ldr	s0, [x1, w25, uxtw #2]
    401c:	1e280928 	fmul	s8, s9, s8
    4020:	1e200908 	fmul	s8, s8, s0
    4024:	b94002fa 	ldr	w26, [x23]
    4028:	1e270340 	fmov	s0, w26
    402c:	aa1703e2 	mov	x2, x23
    4030:	2a1a03e0 	mov	w0, w26
    4034:	1e202900 	fadd	s0, s8, s0
    4038:	1e260001 	fmov	w1, s0
    403c:	940002fd 	bl	4c30 <__aarch64_cas4_acq_rel>
    4040:	6b00035f 	cmp	w26, w0
    4044:	540003a1 	b.ne	40b8 <sptOmpMTTKRP_3D._omp_fn.0+0x188>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    4048:	11000739 	add	w25, w25, #0x1
    404c:	910012f7 	add	x23, x23, #0x4
    4050:	6b19027f 	cmp	w19, w25
    4054:	54fffda1 	b.ne	4008 <sptOmpMTTKRP_3D._omp_fn.0+0xd8>  // b.any
    4058:	f9403be0 	ldr	x0, [sp, #112]
    405c:	910006b5 	add	x21, x21, #0x1
    4060:	eb15001f 	cmp	x0, x21
    4064:	54fffb21 	b.ne	3fc8 <sptOmpMTTKRP_3D._omp_fn.0+0x98>  // b.any
    4068:	a94153f3 	ldp	x19, x20, [sp, #16]
    406c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4070:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4074:	a94573fb 	ldp	x27, x28, [sp, #80]
    4078:	6d4627e8 	ldp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    407c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4080:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4084:	d65f03c0 	ret
    4088:	a94153f3 	ldp	x19, x20, [sp, #16]
    408c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4090:	a94363f7 	ldp	x23, x24, [sp, #48]
    4094:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    4098:	d65f03c0 	ret
    409c:	910006f7 	add	x23, x23, #0x1
    40a0:	d2800000 	mov	x0, #0x0                   	// #0
    40a4:	17ffffb3 	b	3f70 <sptOmpMTTKRP_3D._omp_fn.0+0x40>
    40a8:	a94153f3 	ldp	x19, x20, [sp, #16]
    40ac:	a94363f7 	ldp	x23, x24, [sp, #48]
    40b0:	a94573fb 	ldp	x27, x28, [sp, #80]
    40b4:	17fffff2 	b	407c <sptOmpMTTKRP_3D._omp_fn.0+0x14c>
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    40b8:	2a0003fa 	mov	w26, w0
    40bc:	17ffffdb 	b	4028 <sptOmpMTTKRP_3D._omp_fn.0+0xf8>

00000000000040c0 <sptOmpMTTKRP_3D>:
{
    40c0:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    40c4:	910003fd 	mov	x29, sp
    40c8:	a90153f3 	stp	x19, x20, [sp, #16]
    40cc:	aa0003f4 	mov	x20, x0
    40d0:	aa0103f3 	mov	x19, x1
	sptIndex const stride = mats[0]->stride;
    40d4:	f9400020 	ldr	x0, [x1]
{
    40d8:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptAssert(nmodes ==3);
    40dc:	b9400281 	ldr	w1, [x20]
{
    40e0:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex const stride = mats[0]->stride;
    40e4:	b9400c16 	ldr	w22, [x0, #12]
	sptNnzIndex const nnz = X->nnz;
    40e8:	a9416285 	ldp	x5, x24, [x20, #16]
	sptValue const * const restrict vals = X->values.data;
    40ec:	f9401e97 	ldr	x23, [x20, #56]
	sptAssert(nmodes ==3);
    40f0:	a9046bf9 	stp	x25, x26, [sp, #64]
    40f4:	71000c3f 	cmp	w1, #0x3
    40f8:	54000e61 	b.ne	42c4 <sptOmpMTTKRP_3D+0x204>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    40fc:	f9400e66 	ldr	x6, [x19, #24]
    4100:	aa0203fa 	mov	x26, x2
    4104:	b9400402 	ldr	w2, [x0, #4]
    4108:	b94004c1 	ldr	w1, [x6, #4]
    410c:	6b02003f 	cmp	w1, w2
    4110:	54000ae1 	b.ne	426c <sptOmpMTTKRP_3D+0x1ac>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    4114:	b9400002 	ldr	w2, [x0]
    4118:	b94000a0 	ldr	w0, [x5]
    411c:	6b00005f 	cmp	w2, w0
    4120:	54000bc1 	b.ne	4298 <sptOmpMTTKRP_3D+0x1d8>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4124:	f9400660 	ldr	x0, [x19, #8]
    4128:	b9400402 	ldr	w2, [x0, #4]
    412c:	6b02003f 	cmp	w1, w2
    4130:	540009e1 	b.ne	426c <sptOmpMTTKRP_3D+0x1ac>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    4134:	b9400002 	ldr	w2, [x0]
    4138:	b94004a0 	ldr	w0, [x5, #4]
    413c:	6b00005f 	cmp	w2, w0
    4140:	54000ac1 	b.ne	4298 <sptOmpMTTKRP_3D+0x1d8>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4144:	f9400a60 	ldr	x0, [x19, #16]
    4148:	b9400402 	ldr	w2, [x0, #4]
    414c:	6b01005f 	cmp	w2, w1
    4150:	540008e1 	b.ne	426c <sptOmpMTTKRP_3D+0x1ac>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    4154:	b9400001 	ldr	w1, [x0]
    4158:	b94008a0 	ldr	w0, [x5, #8]
    415c:	6b00003f 	cmp	w1, w0
    4160:	540009c1 	b.ne	4298 <sptOmpMTTKRP_3D+0x1d8>  // b.any
	sptIndex const tmpI = mats[mode]->nrows;
    4164:	f8635a60 	ldr	x0, [x19, w3, uxtw #3]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    4168:	52800315 	mov	w21, #0x18                  	// #24
    416c:	f9401282 	ldr	x2, [x20, #32]
    4170:	a90573fb 	stp	x27, x28, [sp, #80]
    4174:	2a0403f9 	mov	w25, w4
	sptIndex const R = mats[mode]->ncols;
    4178:	29407005 	ldp	w5, w28, [x0]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    417c:	9bb50863 	umaddl	x3, w3, w21, x2
	sptValue * const restrict mvals = mats[nmodes]->values;
    4180:	f94008db 	ldr	x27, [x6, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4184:	52800001 	mov	w1, #0x0                   	// #0
    4188:	1b057ec5 	mul	w5, w22, w5
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    418c:	f9400867 	ldr	x7, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4190:	aa1b03e0 	mov	x0, x27
    4194:	d37ef4a2 	lsl	x2, x5, #2
    4198:	fd0033e8 	str	d8, [sp, #96]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    419c:	f9003fe7 	str	x7, [sp, #120]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    41a0:	97fff444 	bl	12b0 <memset@plt>
	sptIndex times_mat_index_2 = mats_order[2];
    41a4:	29409743 	ldp	w3, w5, [x26, #4]
	sptNewTimer(&timer, 0);
    41a8:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    41ac:	f9401286 	ldr	x6, [x20, #32]
	sptNewTimer(&timer, 0);
    41b0:	910203e0 	add	x0, sp, #0x80
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    41b4:	f8635a64 	ldr	x4, [x19, w3, uxtw #3]
    41b8:	f9003be4 	str	x4, [sp, #112]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    41bc:	9bb51863 	umaddl	x3, w3, w21, x6
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    41c0:	9bb518b5 	umaddl	x21, w5, w21, x6
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    41c4:	f8655a74 	ldr	x20, [x19, w5, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    41c8:	f940087a 	ldr	x26, [x3, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    41cc:	f9400ab3 	ldr	x19, [x21, #16]
	sptNewTimer(&timer, 0);
    41d0:	940000bc 	bl	44c0 <sptNewTimer>
	sptStartTimer(timer);
    41d4:	f94043e0 	ldr	x0, [sp, #128]
    41d8:	940000da 	bl	4540 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    41dc:	a908dff8 	stp	x24, x23, [sp, #136]
    41e0:	2a1903e2 	mov	w2, w25
    41e4:	a9471fe4 	ldp	x4, x7, [sp, #112]
    41e8:	52800003 	mov	w3, #0x0                   	// #0
    41ec:	910223e1 	add	x1, sp, #0x88
    41f0:	f0ffffe0 	adrp	x0, 3000 <sptLoadSparseTensor+0x8fc>
    41f4:	913cc000 	add	x0, x0, #0xf30
    41f8:	a909efe7 	stp	x7, x27, [sp, #152]
    41fc:	a90aebe4 	stp	x4, x26, [sp, #168]
    4200:	a90bcff4 	stp	x20, x19, [sp, #184]
    4204:	291973f6 	stp	w22, w28, [sp, #200]
    4208:	97fff486 	bl	1420 <GOMP_parallel@plt>
		}
	}
	sptStopTimer(timer);
    420c:	f94043e0 	ldr	x0, [sp, #128]
    4210:	940000fc 	bl	4600 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    4214:	f94043e0 	ldr	x0, [sp, #128]
    4218:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    421c:	91166021 	add	x1, x1, #0x598
    4220:	94000138 	bl	4700 <sptPrintElapsedTime>
    4224:	1e604008 	fmov	d8, d0

	sptFreeTimer(timer);
    4228:	f94043e0 	ldr	x0, [sp, #128]
    422c:	94000171 	bl	47f0 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    4230:	1e604100 	fmov	d0, d8
    4234:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4238:	91156000 	add	x0, x0, #0x558
    423c:	97fff471 	bl	1400 <printf@plt>
	printf("\n");
    4240:	52800140 	mov	w0, #0xa                   	// #10
    4244:	97fff47b 	bl	1430 <putchar@plt>

	return 0;
    4248:	a94573fb 	ldp	x27, x28, [sp, #80]
    424c:	52800000 	mov	w0, #0x0                   	// #0
    4250:	fd4033e8 	ldr	d8, [sp, #96]
}
    4254:	a94153f3 	ldp	x19, x20, [sp, #16]
    4258:	a9425bf5 	ldp	x21, x22, [sp, #32]
    425c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4260:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4264:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4268:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    426c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4270:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4274:	91134084 	add	x4, x4, #0x4d0
    4278:	9115c042 	add	x2, x2, #0x570
    427c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4280:	528012c3 	mov	w3, #0x96                  	// #150
    4284:	91166000 	add	x0, x0, #0x598
    4288:	52800041 	mov	w1, #0x2                   	// #2
    428c:	940001d9 	bl	49f0 <spt_ComplainError>
    4290:	52800040 	mov	w0, #0x2                   	// #2
    4294:	17fffff0 	b	4254 <sptOmpMTTKRP_3D+0x194>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4298:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    429c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    42a0:	9114e084 	add	x4, x4, #0x538
    42a4:	9115c042 	add	x2, x2, #0x570
    42a8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    42ac:	52801323 	mov	w3, #0x99                  	// #153
    42b0:	91166000 	add	x0, x0, #0x598
    42b4:	52800041 	mov	w1, #0x2                   	// #2
    42b8:	940001ce 	bl	49f0 <spt_ComplainError>
    42bc:	52800040 	mov	w0, #0x2                   	// #2
    42c0:	17ffffe5 	b	4254 <sptOmpMTTKRP_3D+0x194>
	sptAssert(nmodes ==3);
    42c4:	12800000 	mov	w0, #0xffffffff            	// #-1
    42c8:	a90573fb 	stp	x27, x28, [sp, #80]
    42cc:	fd0033e8 	str	d8, [sp, #96]
    42d0:	97fff3bc 	bl	11c0 <exit@plt>

00000000000042d4 <sptOmpMTTKRP>:
{
    42d4:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    42d8:	910003fd 	mov	x29, sp
    42dc:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const nmodes = X->nmodes;
    42e0:	b9400014 	ldr	w20, [x0]
	if(nmodes == 3) {
    42e4:	71000e9f 	cmp	w20, #0x3
    42e8:	54000da0 	b.eq	449c <sptOmpMTTKRP+0x1c8>  // b.none
	sptNnzIndex const nnz = X->nnz;
    42ec:	a90363f7 	stp	x23, x24, [sp, #48]
    42f0:	aa0003f7 	mov	x23, x0
	sptIndex const * const ndims = X->ndims;
    42f4:	aa0103f3 	mov	x19, x1
    42f8:	a9025bf5 	stp	x21, x22, [sp, #32]
    42fc:	2a0303e9 	mov	w9, w3
    4300:	aa0203f6 	mov	x22, x2
    4304:	2a0403f5 	mov	w21, w4
	sptValue * const restrict mvals = mats[nmodes]->values;
    4308:	f8745820 	ldr	x0, [x1, w20, uxtw #3]
	sptIndex const * const ndims = X->ndims;
    430c:	f9400ae8 	ldr	x8, [x23, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    4310:	340003d4 	cbz	w20, 4388 <sptOmpMTTKRP+0xb4>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4314:	b9400401 	ldr	w1, [x0, #4]
    4318:	d2800005 	mov	x5, #0x0                   	// #0
    431c:	14000008 	b	433c <sptOmpMTTKRP+0x68>
		if(mats[i]->nrows != ndims[i]) {
    4320:	b94000c7 	ldr	w7, [x6]
    4324:	b8657906 	ldr	w6, [x8, x5, lsl #2]
    4328:	6b0600ff 	cmp	w7, w6
    432c:	54000aa1 	b.ne	4480 <sptOmpMTTKRP+0x1ac>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    4330:	910004a5 	add	x5, x5, #0x1
    4334:	6b05029f 	cmp	w20, w5
    4338:	54000289 	b.ls	4388 <sptOmpMTTKRP+0xb4>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    433c:	f8657a66 	ldr	x6, [x19, x5, lsl #3]
    4340:	b94004c7 	ldr	w7, [x6, #4]
    4344:	6b0100ff 	cmp	w7, w1
    4348:	54fffec0 	b.eq	4320 <sptOmpMTTKRP+0x4c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    434c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4350:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4354:	91134084 	add	x4, x4, #0x4d0
    4358:	9115c042 	add	x2, x2, #0x570
    435c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4360:	52800823 	mov	w3, #0x41                  	// #65
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4364:	91166000 	add	x0, x0, #0x598
    4368:	52800041 	mov	w1, #0x2                   	// #2
    436c:	940001a1 	bl	49f0 <spt_ComplainError>
    4370:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4374:	52800040 	mov	w0, #0x2                   	// #2
    4378:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    437c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4380:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    4384:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    4388:	f8695a64 	ldr	x4, [x19, w9, uxtw #3]
    438c:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4390:	52800305 	mov	w5, #0x18                  	// #24
	sptIndex const stride = mats[0]->stride;
    4394:	f9400262 	ldr	x2, [x19]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4398:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    439c:	b940049a 	ldr	w26, [x4, #4]
	sptValue * const restrict mvals = mats[nmodes]->values;
    43a0:	f9400819 	ldr	x25, [x0, #16]
    43a4:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex const stride = mats[0]->stride;
    43a8:	b9400c58 	ldr	w24, [x2, #12]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    43ac:	b9400082 	ldr	w2, [x4]
    43b0:	aa1903e0 	mov	x0, x25
	sptIndex const * const mode_ind = X->inds[mode].data;
    43b4:	a9418ee4 	ldp	x4, x3, [x23, #24]
	sptNnzIndex const nnz = X->nnz;
    43b8:	f9003fe4 	str	x4, [sp, #120]
	sptValue const * const vals = X->values.data;
    43bc:	f9401efc 	ldr	x28, [x23, #56]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    43c0:	1b027f02 	mul	w2, w24, w2
	sptIndex const * const mode_ind = X->inds[mode].data;
    43c4:	9ba50d23 	umaddl	x3, w9, w5, x3
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    43c8:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const mode_ind = X->inds[mode].data;
    43cc:	f940087b 	ldr	x27, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    43d0:	fd0033e8 	str	d8, [sp, #96]
    43d4:	97fff3b7 	bl	12b0 <memset@plt>
	sptNewTimer(&timer, 0);
    43d8:	52800001 	mov	w1, #0x0                   	// #0
    43dc:	910203e0 	add	x0, sp, #0x80
    43e0:	94000038 	bl	44c0 <sptNewTimer>
	sptStartTimer(timer);
    43e4:	f94043e0 	ldr	x0, [sp, #128]
    43e8:	94000056 	bl	4540 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    43ec:	a908cff7 	stp	x23, x19, [sp, #136]
    43f0:	2a1503e2 	mov	w2, w21
    43f4:	f9403fe4 	ldr	x4, [sp, #120]
    43f8:	52800003 	mov	w3, #0x0                   	// #0
    43fc:	910223e1 	add	x1, sp, #0x88
    4400:	f0ffffe0 	adrp	x0, 3000 <sptLoadSparseTensor+0x8fc>
    4404:	91338000 	add	x0, x0, #0xce0
    4408:	a90993f6 	stp	x22, x4, [sp, #152]
    440c:	a90aeffc 	stp	x28, x27, [sp, #168]
    4410:	f9005ff9 	str	x25, [sp, #184]
    4414:	291863f4 	stp	w20, w24, [sp, #192]
    4418:	b900cbfa 	str	w26, [sp, #200]
    441c:	97fff401 	bl	1420 <GOMP_parallel@plt>
	sptStopTimer(timer);
    4420:	f94043e0 	ldr	x0, [sp, #128]
    4424:	94000077 	bl	4600 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    4428:	f94043e0 	ldr	x0, [sp, #128]
    442c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    4430:	91166021 	add	x1, x1, #0x598
    4434:	940000b3 	bl	4700 <sptPrintElapsedTime>
    4438:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    443c:	f94043e0 	ldr	x0, [sp, #128]
    4440:	940000ec 	bl	47f0 <sptFreeTimer>
	printf("[Total time]: %lf\n", total_time);
    4444:	1e604100 	fmov	d0, d8
    4448:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    444c:	91156000 	add	x0, x0, #0x558
    4450:	97fff3ec 	bl	1400 <printf@plt>
	printf("\n");
    4454:	52800140 	mov	w0, #0xa                   	// #10
    4458:	97fff3f6 	bl	1430 <putchar@plt>
	return 0;
    445c:	52800000 	mov	w0, #0x0                   	// #0
    4460:	fd4033e8 	ldr	d8, [sp, #96]
}
    4464:	a94153f3 	ldp	x19, x20, [sp, #16]
	return 0;
    4468:	a9425bf5 	ldp	x21, x22, [sp, #32]
    446c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4470:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4474:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    4478:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    447c:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4480:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4484:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4488:	9114e084 	add	x4, x4, #0x538
    448c:	9115c042 	add	x2, x2, #0x570
    4490:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4494:	52800883 	mov	w3, #0x44                  	// #68
    4498:	17ffffb3 	b	4364 <sptOmpMTTKRP+0x90>
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    449c:	97ffff09 	bl	40c0 <sptOmpMTTKRP_3D>
    44a0:	34fff6e0 	cbz	w0, 437c <sptOmpMTTKRP+0xa8>
    44a4:	12800000 	mov	w0, #0xffffffff            	// #-1
    44a8:	a9025bf5 	stp	x21, x22, [sp, #32]
    44ac:	a90363f7 	stp	x23, x24, [sp, #48]
    44b0:	a9046bf9 	stp	x25, x26, [sp, #64]
    44b4:	a90573fb 	stp	x27, x28, [sp, #80]
    44b8:	fd0033e8 	str	d8, [sp, #96]
    44bc:	97fff341 	bl	11c0 <exit@plt>

00000000000044c0 <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    44c0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    44c4:	910003fd 	mov	x29, sp
    44c8:	a90153f3 	stp	x19, x20, [sp, #16]
    44cc:	aa0003f4 	mov	x20, x0
    44d0:	2a0103f3 	mov	w19, w1
	*timer = (sptTimer) malloc(sizeof **timer);
    44d4:	d2800500 	mov	x0, #0x28                  	// #40
    44d8:	97fff362 	bl	1260 <malloc@plt>
	(*timer)->use_cuda = use_cuda;
    44dc:	b9000013 	str	w19, [x0]
	*timer = (sptTimer) malloc(sizeof **timer);
    44e0:	f9000280 	str	x0, [x20]
	if(use_cuda) {
    44e4:	350000b3 	cbnz	w19, 44f8 <sptNewTimer+0x38>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	return 0;
}
    44e8:	2a1303e0 	mov	w0, w19
    44ec:	a94153f3 	ldp	x19, x20, [sp, #16]
    44f0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    44f4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    44f8:	52800073 	mov	w19, #0x3                   	// #3
    44fc:	52800061 	mov	w1, #0x3                   	// #3
    4500:	72a00053 	movk	w19, #0x2, lsl #16
    4504:	72a00041 	movk	w1, #0x2, lsl #16
    4508:	528004e3 	mov	w3, #0x27                  	// #39
    450c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4510:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4514:	9116c084 	add	x4, x4, #0x5b0
    4518:	91176042 	add	x2, x2, #0x5d8
    451c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4520:	91180000 	add	x0, x0, #0x600
    4524:	94000133 	bl	49f0 <spt_ComplainError>
}
    4528:	2a1303e0 	mov	w0, w19
    452c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4530:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4534:	d65f03c0 	ret
    4538:	d503201f 	nop
    453c:	d503201f 	nop

0000000000004540 <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    4540:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4544:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    4548:	b9400001 	ldr	w1, [x0]
    454c:	350000e1 	cbnz	w1, 4568 <sptStartTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    4550:	91002001 	add	x1, x0, #0x8
    4554:	52800020 	mov	w0, #0x1                   	// #1
    4558:	97fff326 	bl	11f0 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    455c:	35000220 	cbnz	w0, 45a0 <sptStartTimer+0x60>
	}
	return 0;
}
    4560:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4564:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4568:	52800061 	mov	w1, #0x3                   	// #3
    456c:	528005e3 	mov	w3, #0x2f                  	// #47
    4570:	72a00041 	movk	w1, #0x2, lsl #16
    4574:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4578:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    457c:	9116c084 	add	x4, x4, #0x5b0
    4580:	91176042 	add	x2, x2, #0x5d8
    4584:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4588:	91180000 	add	x0, x0, #0x600
    458c:	94000119 	bl	49f0 <spt_ComplainError>
    4590:	52800060 	mov	w0, #0x3                   	// #3
    4594:	72a00040 	movk	w0, #0x2, lsl #16
}
    4598:	a8c27bfd 	ldp	x29, x30, [sp], #32
    459c:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    45a0:	a90153f3 	stp	x19, x20, [sp, #16]
    45a4:	97fff39b 	bl	1410 <__errno_location@plt>
    45a8:	aa0003f3 	mov	x19, x0
	return 0;
    45ac:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    45b0:	b9400274 	ldr	w20, [x19]
    45b4:	3140429f 	cmn	w20, #0x10, lsl #12
    45b8:	54000061 	b.ne	45c4 <sptStartTimer+0x84>  // b.any
    45bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    45c0:	17ffffe8 	b	4560 <sptStartTimer+0x20>
    45c4:	2a1403e0 	mov	w0, w20
    45c8:	97fff352 	bl	1310 <strerror@plt>
    45cc:	11404281 	add	w1, w20, #0x10, lsl #12
    45d0:	aa0003e4 	mov	x4, x0
    45d4:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    45d8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    45dc:	91176042 	add	x2, x2, #0x5d8
    45e0:	91180000 	add	x0, x0, #0x600
    45e4:	52800643 	mov	w3, #0x32                  	// #50
    45e8:	94000102 	bl	49f0 <spt_ComplainError>
    45ec:	b9400260 	ldr	w0, [x19]
    45f0:	a94153f3 	ldp	x19, x20, [sp, #16]
    45f4:	11404000 	add	w0, w0, #0x10, lsl #12
    45f8:	17ffffda 	b	4560 <sptStartTimer+0x20>
    45fc:	d503201f 	nop

0000000000004600 <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    4600:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4604:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    4608:	b9400001 	ldr	w1, [x0]
    460c:	350000e1 	cbnz	w1, 4628 <sptStopTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    4610:	91006001 	add	x1, x0, #0x18
    4614:	52800020 	mov	w0, #0x1                   	// #1
    4618:	97fff2f6 	bl	11f0 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    461c:	35000220 	cbnz	w0, 4660 <sptStopTimer+0x60>
	}
	return 0;
}
    4620:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4624:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4628:	52800061 	mov	w1, #0x3                   	// #3
    462c:	52800743 	mov	w3, #0x3a                  	// #58
    4630:	72a00041 	movk	w1, #0x2, lsl #16
    4634:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4638:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    463c:	9116c084 	add	x4, x4, #0x5b0
    4640:	91176042 	add	x2, x2, #0x5d8
    4644:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4648:	91180000 	add	x0, x0, #0x600
    464c:	940000e9 	bl	49f0 <spt_ComplainError>
    4650:	52800060 	mov	w0, #0x3                   	// #3
    4654:	72a00040 	movk	w0, #0x2, lsl #16
}
    4658:	a8c27bfd 	ldp	x29, x30, [sp], #32
    465c:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    4660:	a90153f3 	stp	x19, x20, [sp, #16]
    4664:	97fff36b 	bl	1410 <__errno_location@plt>
    4668:	aa0003f3 	mov	x19, x0
	return 0;
    466c:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    4670:	b9400274 	ldr	w20, [x19]
    4674:	3140429f 	cmn	w20, #0x10, lsl #12
    4678:	54000061 	b.ne	4684 <sptStopTimer+0x84>  // b.any
    467c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4680:	17ffffe8 	b	4620 <sptStopTimer+0x20>
    4684:	2a1403e0 	mov	w0, w20
    4688:	97fff322 	bl	1310 <strerror@plt>
    468c:	11404281 	add	w1, w20, #0x10, lsl #12
    4690:	aa0003e4 	mov	x4, x0
    4694:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4698:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    469c:	91176042 	add	x2, x2, #0x5d8
    46a0:	91180000 	add	x0, x0, #0x600
    46a4:	528007a3 	mov	w3, #0x3d                  	// #61
    46a8:	940000d2 	bl	49f0 <spt_ComplainError>
    46ac:	b9400260 	ldr	w0, [x19]
    46b0:	a94153f3 	ldp	x19, x20, [sp, #16]
    46b4:	11404000 	add	w0, w0, #0x10, lsl #12
    46b8:	17ffffda 	b	4620 <sptStopTimer+0x20>
    46bc:	d503201f 	nop

00000000000046c0 <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
	if(timer->use_cuda) {
    46c0:	b9400001 	ldr	w1, [x0]
    46c4:	35000161 	cbnz	w1, 46f0 <sptElapsedTime+0x30>
		return NAN;
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    46c8:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    46cc:	a9408c04 	ldp	x4, x3, [x0, #8]
    46d0:	fd431022 	ldr	d2, [x1, #1568]
    46d4:	a9418801 	ldp	x1, x2, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    46d8:	cb040020 	sub	x0, x1, x4
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    46dc:	cb030041 	sub	x1, x2, x3
    46e0:	9e620021 	scvtf	d1, x1
    46e4:	9e620000 	scvtf	d0, x0
    46e8:	1f420020 	fmadd	d0, d1, d2, d0
	}
}
    46ec:	d65f03c0 	ret
		return NAN;
    46f0:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    46f4:	9e670000 	fmov	d0, x0
}
    46f8:	d65f03c0 	ret
    46fc:	d503201f 	nop

0000000000004700 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4700:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4704:	aa0103e2 	mov	x2, x1
    4708:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    470c:	b9400001 	ldr	w1, [x0]
double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4710:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4714:	350002a1 	cbnz	w1, 4768 <sptPrintElapsedTime+0x68>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4718:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    471c:	a9409005 	ldp	x5, x4, [x0, #8]
    4720:	fd431021 	ldr	d1, [x1, #1568]
    4724:	a9418403 	ldp	x3, x1, [x0, #24]
    4728:	cb040021 	sub	x1, x1, x4
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    472c:	cb050060 	sub	x0, x3, x5
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4730:	9e620020 	scvtf	d0, x1
    4734:	9e620008 	scvtf	d8, x0
    4738:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer);
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    473c:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    4740:	1e604100 	fmov	d0, d8
    4744:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    4748:	91184021 	add	x1, x1, #0x610
    474c:	f947e000 	ldr	x0, [x0, #4032]
    4750:	f9400000 	ldr	x0, [x0]
    4754:	97fff33b 	bl	1440 <fprintf@plt>
	return elapsed_time;
}
    4758:	1e604100 	fmov	d0, d8
    475c:	fd400be8 	ldr	d8, [sp, #16]
    4760:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4764:	d65f03c0 	ret
		return NAN;
    4768:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    476c:	9e670008 	fmov	d8, x0
    4770:	17fffff3 	b	473c <sptPrintElapsedTime+0x3c>

0000000000004774 <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4774:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4778:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    477c:	b9400003 	ldr	w3, [x0]
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    4780:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4784:	350002e3 	cbnz	w3, 47e0 <sptPrintAverageElapsedTime+0x6c>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4788:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x2f0>
    478c:	a9409406 	ldp	x6, x5, [x0, #8]
    4790:	fd431061 	ldr	d1, [x3, #1568]
    4794:	a9419003 	ldp	x3, x4, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4798:	cb060060 	sub	x0, x3, x6
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    479c:	cb050083 	sub	x3, x4, x5
    47a0:	9e620060 	scvtf	d0, x3
    47a4:	9e620008 	scvtf	d8, x0
    47a8:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer) / niters;
    47ac:	1e620020 	scvtf	d0, w1
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    47b0:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    47b4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    47b8:	91184021 	add	x1, x1, #0x610
    47bc:	f947e000 	ldr	x0, [x0, #4032]
	double elapsed_time = sptElapsedTime(timer) / niters;
    47c0:	1e601908 	fdiv	d8, d8, d0
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    47c4:	f9400000 	ldr	x0, [x0]
    47c8:	1e604100 	fmov	d0, d8
    47cc:	97fff31d 	bl	1440 <fprintf@plt>
	return elapsed_time;
}
    47d0:	1e604100 	fmov	d0, d8
    47d4:	fd400be8 	ldr	d8, [sp, #16]
    47d8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    47dc:	d65f03c0 	ret
		return NAN;
    47e0:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    47e4:	9e670008 	fmov	d8, x0
    47e8:	17fffff1 	b	47ac <sptPrintAverageElapsedTime+0x38>
    47ec:	d503201f 	nop

00000000000047f0 <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    47f0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    47f4:	910003fd 	mov	x29, sp
    47f8:	f9000bf3 	str	x19, [sp, #16]
	if(timer->use_cuda) {
    47fc:	b9400013 	ldr	w19, [x0]
    4800:	350000d3 	cbnz	w19, 4818 <sptFreeTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	free(timer);
    4804:	97fff2e7 	bl	13a0 <free@plt>
	return 0;
}
    4808:	2a1303e0 	mov	w0, w19
    480c:	f9400bf3 	ldr	x19, [sp, #16]
    4810:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4814:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4818:	52800061 	mov	w1, #0x3                   	// #3
    481c:	52800073 	mov	w19, #0x3                   	// #3
    4820:	72a00041 	movk	w1, #0x2, lsl #16
    4824:	52800b63 	mov	w3, #0x5b                  	// #91
    4828:	72a00053 	movk	w19, #0x2, lsl #16
    482c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x2f0>
    4830:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4834:	9116c084 	add	x4, x4, #0x5b0
    4838:	91176042 	add	x2, x2, #0x5d8
    483c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4840:	91180000 	add	x0, x0, #0x600
    4844:	9400006b 	bl	49f0 <spt_ComplainError>
}
    4848:	2a1303e0 	mov	w0, w19
    484c:	f9400bf3 	ldr	x19, [sp, #16]
    4850:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4854:	d65f03c0 	ret
    4858:	d503201f 	nop
    485c:	d503201f 	nop

0000000000004860 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4860:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
    4864:	910003fd 	mov	x29, sp
    4868:	a9025bf5 	stp	x21, x22, [sp, #32]
    486c:	aa0003f6 	mov	x22, x0
    4870:	aa0103f5 	mov	x21, x1
    4874:	a9046bf9 	stp	x25, x26, [sp, #64]
	int iores;
	sptIndex nrows = mtx->nrows;
	sptIndex ncols = mtx->ncols;
	sptIndex stride = mtx->stride;
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4878:	aa0103e0 	mov	x0, x1
    487c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
	sptIndex ncols = mtx->ncols;
    4880:	294066da 	ldp	w26, w25, [x22]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    4884:	9118a021 	add	x1, x1, #0x628
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4888:	a90153f3 	stp	x19, x20, [sp, #16]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    488c:	2a1a03e2 	mov	w2, w26
    4890:	2a1903e3 	mov	w3, w25
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    4894:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex stride = mtx->stride;
    4898:	b9400edb 	ldr	w27, [x22, #12]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    489c:	97fff2e9 	bl	1440 <fprintf@plt>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    48a0:	37f80820 	tbnz	w0, #31, 49a4 <sptDumpMatrix+0x144>
	for(sptIndex i=0; i < nrows; ++i) {
    48a4:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex ncols = mtx->ncols;
    48a8:	2a1903f4 	mov	w20, w25
    48ac:	b0000017 	adrp	x23, 5000 <_IO_stdin_used+0x2f0>
	for(sptIndex i=0; i < nrows; ++i) {
    48b0:	52800018 	mov	w24, #0x0                   	// #0
		for(sptIndex j=0; j < ncols; ++j) {
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    48b4:	9119c2f7 	add	x23, x23, #0x670
	for(sptIndex i=0; i < nrows; ++i) {
    48b8:	3400061a 	cbz	w26, 4978 <sptDumpMatrix+0x118>
    48bc:	d503201f 	nop
		for(sptIndex j=0; j < ncols; ++j) {
    48c0:	4b190293 	sub	w19, w20, w25
    48c4:	35000099 	cbnz	w25, 48d4 <sptDumpMatrix+0x74>
    48c8:	14000025 	b	495c <sptDumpMatrix+0xfc>
    48cc:	6b13029f 	cmp	w20, w19
    48d0:	54000460 	b.eq	495c <sptDumpMatrix+0xfc>  // b.none
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    48d4:	f9400ac2 	ldr	x2, [x22, #16]
    48d8:	aa1703e1 	mov	x1, x23
    48dc:	aa1503e0 	mov	x0, x21
    48e0:	bc735840 	ldr	s0, [x2, w19, uxtw #2]
		for(sptIndex j=0; j < ncols; ++j) {
    48e4:	11000673 	add	w19, w19, #0x1
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    48e8:	1e22c000 	fcvt	d0, s0
    48ec:	97fff2d5 	bl	1440 <fprintf@plt>
			spt_CheckOSError(iores < 0, "Mtx Dump");
    48f0:	36fffee0 	tbz	w0, #31, 48cc <sptDumpMatrix+0x6c>
    48f4:	97fff2c7 	bl	1410 <__errno_location@plt>
    48f8:	b9400001 	ldr	w1, [x0]
    48fc:	aa0003fc 	mov	x28, x0
    4900:	3140403f 	cmn	w1, #0x10, lsl #12
    4904:	54fffe40 	b.eq	48cc <sptDumpMatrix+0x6c>  // b.none
    4908:	2a0103e0 	mov	w0, w1
    490c:	b9006fe1 	str	w1, [sp, #108]
    4910:	97fff280 	bl	1310 <strerror@plt>
    4914:	aa0003e4 	mov	x4, x0
    4918:	b9406fe1 	ldr	w1, [sp, #108]
    491c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    4920:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    4924:	9118e042 	add	x2, x2, #0x638
    4928:	91198000 	add	x0, x0, #0x660
    492c:	11404021 	add	w1, w1, #0x10, lsl #12
    4930:	528005c3 	mov	w3, #0x2e                  	// #46
    4934:	9400002f 	bl	49f0 <spt_ComplainError>
    4938:	a94363f7 	ldp	x23, x24, [sp, #48]
    493c:	b9400380 	ldr	w0, [x28]
    4940:	11404000 	add	w0, w0, #0x10, lsl #12
		}
		iores = fprintf(fp, "\n");
	}
	iores = fprintf(fp, "\n");
	return 0;
}
    4944:	a94153f3 	ldp	x19, x20, [sp, #16]
    4948:	a9425bf5 	ldp	x21, x22, [sp, #32]
    494c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4950:	a94573fb 	ldp	x27, x28, [sp, #80]
    4954:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4958:	d65f03c0 	ret
		iores = fprintf(fp, "\n");
    495c:	aa1503e1 	mov	x1, x21
    4960:	52800140 	mov	w0, #0xa                   	// #10
	for(sptIndex i=0; i < nrows; ++i) {
    4964:	11000718 	add	w24, w24, #0x1
		iores = fprintf(fp, "\n");
    4968:	97fff21e 	bl	11e0 <fputc@plt>
	for(sptIndex i=0; i < nrows; ++i) {
    496c:	0b1b0294 	add	w20, w20, w27
    4970:	6b18035f 	cmp	w26, w24
    4974:	54fffa61 	b.ne	48c0 <sptDumpMatrix+0x60>  // b.any
	iores = fprintf(fp, "\n");
    4978:	aa1503e1 	mov	x1, x21
    497c:	52800140 	mov	w0, #0xa                   	// #10
    4980:	97fff218 	bl	11e0 <fputc@plt>
	return 0;
    4984:	52800000 	mov	w0, #0x0                   	// #0
}
    4988:	a94153f3 	ldp	x19, x20, [sp, #16]
    498c:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    4990:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    4994:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4998:	a94573fb 	ldp	x27, x28, [sp, #80]
    499c:	a8c77bfd 	ldp	x29, x30, [sp], #112
    49a0:	d65f03c0 	ret
	spt_CheckOSError(iores < 0, "Mtx Dump");
    49a4:	97fff29b 	bl	1410 <__errno_location@plt>
    49a8:	b9400014 	ldr	w20, [x0]
    49ac:	aa0003f3 	mov	x19, x0
    49b0:	3140429f 	cmn	w20, #0x10, lsl #12
    49b4:	54fff780 	b.eq	48a4 <sptDumpMatrix+0x44>  // b.none
    49b8:	2a1403e0 	mov	w0, w20
    49bc:	97fff255 	bl	1310 <strerror@plt>
    49c0:	11404281 	add	w1, w20, #0x10, lsl #12
    49c4:	aa0003e4 	mov	x4, x0
    49c8:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x2f0>
    49cc:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x2f0>
    49d0:	9118e042 	add	x2, x2, #0x638
    49d4:	91198000 	add	x0, x0, #0x660
    49d8:	52800543 	mov	w3, #0x2a                  	// #42
    49dc:	94000005 	bl	49f0 <spt_ComplainError>
    49e0:	b9400260 	ldr	w0, [x19]
    49e4:	11404000 	add	w0, w0, #0x10, lsl #12
    49e8:	17ffffd7 	b	4944 <sptDumpMatrix+0xe4>
    49ec:	d503201f 	nop

00000000000049f0 <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    49f0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    49f4:	910003fd 	mov	x29, sp
    49f8:	a90153f3 	stp	x19, x20, [sp, #16]
	g_last_error.errcode = errcode;
    49fc:	d53bd053 	mrs	x19, tpidr_el0
    4a00:	91400273 	add	x19, x19, #0x0, lsl #12
    4a04:	91004273 	add	x19, x19, #0x10
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4a08:	a9025bf5 	stp	x21, x22, [sp, #32]
    4a0c:	aa0003f4 	mov	x20, x0
    4a10:	2a0103f5 	mov	w21, w1
    4a14:	a90363f7 	stp	x23, x24, [sp, #48]
    4a18:	aa0203f6 	mov	x22, x2
    4a1c:	2a0303f7 	mov	w23, w3
	g_last_error.module = module;
    4a20:	f9000260 	str	x0, [x19]
	g_last_error.errcode = errcode;
    4a24:	b9000a61 	str	w1, [x19, #8]
	g_last_error.file = file;
    4a28:	f9000a62 	str	x2, [x19, #16]
	g_last_error.line = line;
    4a2c:	b9001a63 	str	w3, [x19, #24]
	if(reason) {
		free(g_last_error.reason);
    4a30:	f9401266 	ldr	x6, [x19, #32]
	if(reason) {
    4a34:	b4000324 	cbz	x4, 4a98 <spt_ComplainError+0xa8>
		free(g_last_error.reason);
    4a38:	aa0603e0 	mov	x0, x6
    4a3c:	aa0403f8 	mov	x24, x4
    4a40:	97fff258 	bl	13a0 <free@plt>
		g_last_error.reason = strdup(reason);
    4a44:	aa1803e0 	mov	x0, x24
    4a48:	97fff22a 	bl	12f0 <strdup@plt>
    4a4c:	f9001260 	str	x0, [x19, #32]
    4a50:	aa0003e6 	mov	x6, x0
		if(!g_last_error.reason) {
    4a54:	b4000440 	cbz	x0, 4adc <spt_ComplainError+0xec>
			abort();
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4a58:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4a5c:	394000c1 	ldrb	w1, [x6]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4a60:	f947d800 	ldr	x0, [x0, #4016]
    4a64:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4a68:	35000241 	cbnz	w1, 4ab0 <spt_ComplainError+0xc0>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4a6c:	2a1703e5 	mov	w5, w23
    4a70:	aa1603e4 	mov	x4, x22
    4a74:	2a1503e3 	mov	w3, w21
    4a78:	aa1403e2 	mov	x2, x20
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    4a7c:	a94153f3 	ldp	x19, x20, [sp, #16]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4a80:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
}
    4a84:	a9425bf5 	ldp	x21, x22, [sp, #32]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4a88:	911a6021 	add	x1, x1, #0x698
}
    4a8c:	a94363f7 	ldp	x23, x24, [sp, #48]
    4a90:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    4a94:	17fff26b 	b	1440 <fprintf@plt>
    4a98:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    4a9c:	f947d800 	ldr	x0, [x0, #4016]
    4aa0:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    4aa4:	b4fffe46 	cbz	x6, 4a6c <spt_ComplainError+0x7c>
    4aa8:	394000c1 	ldrb	w1, [x6]
    4aac:	34fffe01 	cbz	w1, 4a6c <spt_ComplainError+0x7c>
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4ab0:	2a1703e5 	mov	w5, w23
    4ab4:	aa1603e4 	mov	x4, x22
    4ab8:	2a1503e3 	mov	w3, w21
    4abc:	aa1403e2 	mov	x2, x20
}
    4ac0:	a94153f3 	ldp	x19, x20, [sp, #16]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4ac4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
}
    4ac8:	a9425bf5 	ldp	x21, x22, [sp, #32]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4acc:	9119e021 	add	x1, x1, #0x678
}
    4ad0:	a94363f7 	ldp	x23, x24, [sp, #48]
    4ad4:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4ad8:	17fff25a 	b	1440 <fprintf@plt>
			abort();
    4adc:	97fff21d 	bl	1350 <abort@plt>

0000000000004ae0 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    4ae0:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
	double size = (double)bytes;
	int suff = 0;
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
	while(size > 1024 && suff < 5) {
    4ae4:	d2e81201 	mov	x1, #0x4090000000000000    	// #4652218415073722368
    4ae8:	9e670021 	fmov	d1, x1
{
    4aec:	910003fd 	mov	x29, sp
    4af0:	fd000be8 	str	d8, [sp, #16]
    4af4:	9e670008 	fmov	d8, x0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4af8:	f0000080 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    4afc:	910a0000 	add	x0, x0, #0x280
	double size = (double)bytes;
    4b00:	7e61d900 	ucvtf	d0, d8
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4b04:	9100a3e1 	add	x1, sp, #0x28
    4b08:	a9400c02 	ldp	x2, x3, [x0]
    4b0c:	a9028fe2 	stp	x2, x3, [sp, #40]
	while(size > 1024 && suff < 5) {
    4b10:	1e612010 	fcmpe	d0, d1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4b14:	a9410c02 	ldp	x2, x3, [x0, #16]
    4b18:	a9010c22 	stp	x2, x3, [x1, #16]
    4b1c:	f9401000 	ldr	x0, [x0, #32]
    4b20:	f9001020 	str	x0, [x1, #32]
	while(size > 1024 && suff < 5) {
    4b24:	5400006c 	b.gt	4b30 <sptBytesString+0x50>
	int suff = 0;
    4b28:	52800000 	mov	w0, #0x0                   	// #0
    4b2c:	14000019 	b	4b90 <sptBytesString+0xb0>
		size /= 1024.;
    4b30:	d2e7ea00 	mov	x0, #0x3f50000000000000    	// #4562146422526312448
    4b34:	9e670002 	fmov	d2, x0
    4b38:	1e620800 	fmul	d0, d0, d2
	while(size > 1024 && suff < 5) {
    4b3c:	1e612010 	fcmpe	d0, d1
    4b40:	5400006c 	b.gt	4b4c <sptBytesString+0x6c>
		++suff;
    4b44:	52800020 	mov	w0, #0x1                   	// #1
    4b48:	14000012 	b	4b90 <sptBytesString+0xb0>
		size /= 1024.;
    4b4c:	1e620800 	fmul	d0, d0, d2
	while(size > 1024 && suff < 5) {
    4b50:	1e612010 	fcmpe	d0, d1
    4b54:	5400006c 	b.gt	4b60 <sptBytesString+0x80>
		++suff;
    4b58:	52800040 	mov	w0, #0x2                   	// #2
    4b5c:	1400000d 	b	4b90 <sptBytesString+0xb0>
		size /= 1024.;
    4b60:	1e620800 	fmul	d0, d0, d2
	while(size > 1024 && suff < 5) {
    4b64:	1e612010 	fcmpe	d0, d1
    4b68:	5400006c 	b.gt	4b74 <sptBytesString+0x94>
		++suff;
    4b6c:	52800060 	mov	w0, #0x3                   	// #3
    4b70:	14000008 	b	4b90 <sptBytesString+0xb0>
		size /= 1024.;
    4b74:	1e620800 	fmul	d0, d0, d2
	while(size > 1024 && suff < 5) {
    4b78:	1e612010 	fcmpe	d0, d1
    4b7c:	5400006c 	b.gt	4b88 <sptBytesString+0xa8>
		++suff;
    4b80:	52800080 	mov	w0, #0x4                   	// #4
    4b84:	14000003 	b	4b90 <sptBytesString+0xb0>
		size /= 1024.;
    4b88:	1e620800 	fmul	d0, d0, d2
		++suff;
    4b8c:	528000a0 	mov	w0, #0x5                   	// #5
	}
	char * ret = NULL;
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4b90:	f860d822 	ldr	x2, [x1, w0, sxtw #3]
    4b94:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    4b98:	910083e0 	add	x0, sp, #0x20
    4b9c:	911b8021 	add	x1, x1, #0x6e0
	char * ret = NULL;
    4ba0:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4ba4:	97fff197 	bl	1200 <asprintf@plt>
    4ba8:	3100041f 	cmn	w0, #0x1
    4bac:	540000a0 	b.eq	4bc0 <sptBytesString+0xe0>  // b.none
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
		ret = NULL;
	}
	return ret;
}
    4bb0:	fd400be8 	ldr	d8, [sp, #16]
	return ret;
    4bb4:	f94013e0 	ldr	x0, [sp, #32]
}
    4bb8:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4bbc:	d65f03c0 	ret
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    4bc0:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0xfb88>
    4bc4:	9e660102 	fmov	x2, d8
    4bc8:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x2f0>
    4bcc:	911ba021 	add	x1, x1, #0x6e8
    4bd0:	f947d800 	ldr	x0, [x0, #4016]
    4bd4:	f9400000 	ldr	x0, [x0]
    4bd8:	97fff21a 	bl	1440 <fprintf@plt>
    4bdc:	d2800000 	mov	x0, #0x0                   	// #0
}
    4be0:	fd400be8 	ldr	d8, [sp, #16]
    4be4:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4be8:	d65f03c0 	ret
    4bec:	d503201f 	nop

0000000000004bf0 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    4bf0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4bf4:	910003fd 	mov	x29, sp
    4bf8:	fd000be8 	str	d8, [sp, #16]
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    4bfc:	97fff1b9 	bl	12e0 <rand@plt>
    4c00:	1e028408 	scvtf	s8, w0, #31
    4c04:	1e211000 	fmov	s0, #3.000000000000000000e+00
    4c08:	1e200908 	fmul	s8, s8, s0
	if(rand() % 2 == 0) {
    4c0c:	97fff1b5 	bl	12e0 <rand@plt>
		v *= -1;
    4c10:	1e214100 	fneg	s0, s8
    4c14:	f240001f 	tst	x0, #0x1
	}
	return v;
}
    4c18:	1e280c00 	fcsel	s0, s0, s8, eq  // eq = none
    4c1c:	fd400be8 	ldr	d8, [sp, #16]
    4c20:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4c24:	d65f03c0 	ret
    4c28:	d503201f 	nop
    4c2c:	d503201f 	nop

0000000000004c30 <__aarch64_cas4_acq_rel>:
    4c30:	d503245f 	bti	c
    4c34:	f0000090 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    4c38:	394aa610 	ldrb	w16, [x16, #681]
    4c3c:	34000070 	cbz	w16, 4c48 <__aarch64_cas4_acq_rel+0x18>
    4c40:	88e0fc41 	casal	w0, w1, [x2]
    4c44:	d65f03c0 	ret
    4c48:	2a0003f0 	mov	w16, w0
    4c4c:	885ffc40 	ldaxr	w0, [x2]
    4c50:	6b10001f 	cmp	w0, w16
    4c54:	54000061 	b.ne	4c60 <__aarch64_cas4_acq_rel+0x30>  // b.any
    4c58:	8811fc41 	stlxr	w17, w1, [x2]
    4c5c:	35ffff91 	cbnz	w17, 4c4c <__aarch64_cas4_acq_rel+0x1c>
    4c60:	d65f03c0 	ret
    4c64:	d503201f 	nop
    4c68:	d503201f 	nop
    4c6c:	d503201f 	nop

0000000000004c70 <__libc_csu_init>:
    4c70:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4c74:	910003fd 	mov	x29, sp
    4c78:	a90153f3 	stp	x19, x20, [sp, #16]
    4c7c:	d0000094 	adrp	x20, 16000 <__FRAME_END__+0xfb88>
    4c80:	9134e294 	add	x20, x20, #0xd38
    4c84:	a9025bf5 	stp	x21, x22, [sp, #32]
    4c88:	d0000095 	adrp	x21, 16000 <__FRAME_END__+0xfb88>
    4c8c:	9134a2b5 	add	x21, x21, #0xd28
    4c90:	cb150294 	sub	x20, x20, x21
    4c94:	2a0003f6 	mov	w22, w0
    4c98:	a90363f7 	stp	x23, x24, [sp, #48]
    4c9c:	aa0103f7 	mov	x23, x1
    4ca0:	aa0203f8 	mov	x24, x2
    4ca4:	9343fe94 	asr	x20, x20, #3
    4ca8:	97fff134 	bl	1178 <_init>
    4cac:	b4000174 	cbz	x20, 4cd8 <__libc_csu_init+0x68>
    4cb0:	d2800013 	mov	x19, #0x0                   	// #0
    4cb4:	d503201f 	nop
    4cb8:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    4cbc:	aa1803e2 	mov	x2, x24
    4cc0:	91000673 	add	x19, x19, #0x1
    4cc4:	aa1703e1 	mov	x1, x23
    4cc8:	2a1603e0 	mov	w0, w22
    4ccc:	d63f0060 	blr	x3
    4cd0:	eb13029f 	cmp	x20, x19
    4cd4:	54ffff21 	b.ne	4cb8 <__libc_csu_init+0x48>  // b.any
    4cd8:	a94153f3 	ldp	x19, x20, [sp, #16]
    4cdc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4ce0:	a94363f7 	ldp	x23, x24, [sp, #48]
    4ce4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    4ce8:	d65f03c0 	ret
    4cec:	d503201f 	nop

0000000000004cf0 <__libc_csu_fini>:
    4cf0:	d65f03c0 	ret

Disassembly of section .fini:

0000000000004cf4 <_fini>:
    4cf4:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    4cf8:	910003fd 	mov	x29, sp
    4cfc:	a8c17bfd 	ldp	x29, x30, [sp], #16
    4d00:	d65f03c0 	ret
