
mttkrp:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000001178 <_init>:
    1178:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    117c:	910003fd 	mov	x29, sp
    1180:	9400026b 	bl	1b2c <call_weak_fn>
    1184:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1188:	d65f03c0 	ret

Disassembly of section .plt:

0000000000001190 <.plt>:
    1190:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
    1194:	b00000b0 	adrp	x16, 16000 <__FRAME_END__+0x10088>
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
    1470:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1474:	528000a1 	mov	w1, #0x5                   	// #5
    1478:	91292000 	add	x0, x0, #0xa48
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
    149c:	540030ed 	b.le	1ab8 <main+0x668>
			case 'i':
				strcpy(fname, optarg);
				printf("input file: %s\n", fname); fflush(stdout);
				break;
			case 'o':
				fo = fopen(optarg, "w");
    14a0:	b00000ba 	adrp	x26, 16000 <__FRAME_END__+0x10088>
    14a4:	d00000b7 	adrp	x23, 17000 <memcpy@GLIBC_2.17>
    14a8:	9000003c 	adrp	x28, 5000 <_IO_stdin_used+0x7d8>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14ac:	910582f7 	add	x23, x23, #0x160
				fo = fopen(optarg, "w");
    14b0:	f947df5b 	ldr	x27, [x26, #4024]
    14b4:	9104239c 	add	x28, x28, #0x108
    14b8:	9101f3f9 	add	x25, sp, #0x7c
	bool random = true;
    14bc:	52800035 	mov	w21, #0x1                   	// #1
	FILE *fo = NULL;
    14c0:	d2800016 	mov	x22, #0x0                   	// #0
    14c4:	f0000018 	adrp	x24, 4000 <sptNewTimer>
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    14c8:	aa1903e4 	mov	x4, x25
    14cc:	aa1703e3 	mov	x3, x23
    14d0:	91296302 	add	x2, x24, #0xa58
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
    1510:	f0000001 	adrp	x1, 4000 <sptNewTimer>
    1514:	912a4021 	add	x1, x1, #0xa90
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
    1534:	f0000001 	adrp	x1, 4000 <sptNewTimer>
    1538:	912b0021 	add	x1, x1, #0xac0
    153c:	f9400000 	ldr	x0, [x0]
    1540:	97ffffac 	bl	13f0 <__isoc99_sscanf@plt>
				break;
    1544:	17ffffe1 	b	14c8 <main+0x78>
		switch(c) {
    1548:	7101901f 	cmp	w0, #0x64
    154c:	54002b61 	b.ne	1ab8 <main+0x668>  // b.any
				sscanf(optarg, "%d", &dev_id);
    1550:	f947df40 	ldr	x0, [x26, #4024]
    1554:	9101e3e2 	add	x2, sp, #0x78
    1558:	f0000001 	adrp	x1, 4000 <sptNewTimer>
    155c:	912a6021 	add	x1, x1, #0xa98
    1560:	f9400000 	ldr	x0, [x0]
    1564:	97ffffa3 	bl	13f0 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    1568:	b9407be0 	ldr	w0, [sp, #120]
    156c:	11000800 	add	w0, w0, #0x2
    1570:	7100041f 	cmp	w0, #0x1
    1574:	54fffaa9 	b.ls	14c8 <main+0x78>  // b.plast
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    1578:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0x10088>
    157c:	d2800382 	mov	x2, #0x1c                  	// #28
    1580:	d2800021 	mov	x1, #0x1                   	// #1
    1584:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1588:	f947d863 	ldr	x3, [x3, #4016]
    158c:	912a8000 	add	x0, x0, #0xaa0
    1590:	f9400063 	ldr	x3, [x3]
    1594:	97ffff87 	bl	13b0 <fwrite@plt>
					exit(1);
    1598:	52800020 	mov	w0, #0x1                   	// #1
    159c:	97ffff09 	bl	11c0 <exit@plt>
		switch(c) {
    15a0:	7101d81f 	cmp	w0, #0x76
    15a4:	540028a1 	b.ne	1ab8 <main+0x668>  // b.any
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
    15c8:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    15cc:	912b2000 	add	x0, x0, #0xac8
				printf("output file: %s\n", optarg); fflush(stdout);
    15d0:	97ffff8c 	bl	1400 <printf@plt>
    15d4:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x10088>
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
    1604:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1608:	9129a000 	add	x0, x0, #0xa68
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
    1634:	b4001816 	cbz	x22, 1934 <main+0x4e4>
				printf("output file: %s\n", optarg); fflush(stdout);
    1638:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    163c:	9129e000 	add	x0, x0, #0xa78
    1640:	17ffffe4 	b	15d0 <main+0x180>
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1644:	b94073e1 	ldr	w1, [sp, #112]
    1648:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    164c:	912ba000 	add	x0, x0, #0xae8
	printf("dev_id: %d\n", dev_id);

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1650:	910223f9 	add	x25, sp, #0x88
	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    1654:	97ffff6b 	bl	1400 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    1658:	b9407be1 	ldr	w1, [sp, #120]
    165c:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1660:	912be000 	add	x0, x0, #0xaf8
    1664:	97ffff67 	bl	1400 <printf@plt>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1668:	aa1903e0 	mov	x0, x25
    166c:	910323e2 	add	x2, sp, #0xc8
    1670:	52800021 	mov	w1, #0x1                   	// #1
    1674:	940003fb 	bl	2660 <sptLoadSparseTensor>
    1678:	350015e0 	cbnz	w0, 1934 <main+0x4e4>
	sptSparseTensorStatus(&X, stdout);
    167c:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0x10088>
    1680:	aa1903e0 	mov	x0, x25
    1684:	f947e021 	ldr	x1, [x1, #4032]
    1688:	f9400021 	ldr	x1, [x1]
    168c:	940006a9 	bl	3130 <sptSparseTensorStatus>

	sptIndex nmodes = X.nmodes;
    1690:	b9408bf3 	ldr	w19, [sp, #136]
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    1694:	11000660 	add	w0, w19, #0x1
    1698:	1100067a 	add	w26, w19, #0x1
    169c:	d37df000 	lsl	x0, x0, #3
    16a0:	97fffef0 	bl	1260 <malloc@plt>
    16a4:	aa0003f4 	mov	x20, x0
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
    16c4:	eb17031f 	cmp	x24, x23
    16c8:	54ffff81 	b.ne	16b8 <main+0x268>  // b.any
	}
	sptIndex max_ndims = 0;
    16cc:	52800018 	mov	w24, #0x0                   	// #0
	for(sptIndex m=0; m<nmodes; ++m) {
    16d0:	340002b3 	cbz	w19, 1724 <main+0x2d4>
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    16d4:	f9404fe0 	ldr	x0, [sp, #152]
    16d8:	d2800017 	mov	x23, #0x0                   	// #0
    16dc:	52800018 	mov	w24, #0x0                   	// #0
    16e0:	b8777801 	ldr	w1, [x0, x23, lsl #2]
    16e4:	d37ef6fa 	lsl	x26, x23, #2
    16e8:	b94077e2 	ldr	w2, [sp, #116]
    16ec:	f8777a80 	ldr	x0, [x20, x23, lsl #3]
    16f0:	940005e4 	bl	2e80 <sptNewMatrix>
    16f4:	35001200 	cbnz	w0, 1934 <main+0x4e4>
		// sptAssert(sptConstantMatrix(U[m], 1) == 0);
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    16f8:	f8777a80 	ldr	x0, [x20, x23, lsl #3]
    16fc:	2a1503e1 	mov	w1, w21
    1700:	94000610 	bl	2f40 <sptRandomizeMatrix>
    1704:	35001180 	cbnz	w0, 1934 <main+0x4e4>
		if(X.ndims[m] > max_ndims)
    1708:	f9404fe0 	ldr	x0, [sp, #152]
	for(sptIndex m=0; m<nmodes; ++m) {
    170c:	910006f7 	add	x23, x23, #0x1
    1710:	b87a6801 	ldr	w1, [x0, x26]
    1714:	6b01031f 	cmp	w24, w1
    1718:	1a812318 	csel	w24, w24, w1, cs  // cs = hs, nlast
    171c:	6b17027f 	cmp	w19, w23
    1720:	54fffe08 	b.hi	16e0 <main+0x290>  // b.pmore
			max_ndims = X.ndims[m];
	}
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1724:	2a1303fb 	mov	w27, w19
    1728:	d37d7e60 	ubfiz	x0, x19, #3, #32
    172c:	8b00029a 	add	x26, x20, x0
    1730:	b94077e2 	ldr	w2, [sp, #116]
    1734:	f90037e0 	str	x0, [sp, #104]
    1738:	2a1803e1 	mov	w1, w24
    173c:	f87b7a80 	ldr	x0, [x20, x27, lsl #3]
    1740:	940005d0 	bl	2e80 <sptNewMatrix>
    1744:	35000f80 	cbnz	w0, 1934 <main+0x4e4>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1748:	0f000400 	movi	v0.2s, #0x0
    174c:	f87b7a80 	ldr	x0, [x20, x27, lsl #3]
    1750:	94000635 	bl	3024 <sptConstantMatrix>
    1754:	35000f00 	cbnz	w0, 1934 <main+0x4e4>
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
    1770:	7100067f 	cmp	w19, #0x1
    1774:	54000109 	b.ls	1794 <main+0x344>  // b.plast
		mats_order[i] = (mode+i) % nmodes;
    1778:	0b000062 	add	w2, w3, w0
    177c:	1ad30841 	udiv	w1, w2, w19
    1780:	1b138821 	msub	w1, w1, w19, w2
    1784:	b8207b01 	str	w1, [x24, x0, lsl #2]
	for(sptIndex i=1; i<nmodes; ++i)
    1788:	91000400 	add	x0, x0, #0x1
    178c:	6b00027f 	cmp	w19, w0
    1790:	54ffff48 	b.hi	1778 <main+0x328>  // b.pmore

	/* For warm-up caches, timing not included */
	if(dev_id == -2) {
    1794:	b9407be0 	ldr	w0, [sp, #120]
    1798:	3100081f 	cmn	w0, #0x2
    179c:	540012c0 	b.eq	19f4 <main+0x5a4>  // b.none
		nthreads = 1;
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
	} else if(dev_id == -1) {
    17a0:	3100041f 	cmn	w0, #0x1
    17a4:	54001340 	b.eq	1a0c <main+0x5bc>  // b.none
    17a8:	910203f7 	add	x23, sp, #0x80
		nthreads = 1;
    17ac:	5280003c 	mov	w28, #0x1                   	// #1
#endif
	}


	sptTimer timer;
	sptNewTimer(&timer, 0);
    17b0:	aa1703e0 	mov	x0, x23
    17b4:	52800001 	mov	w1, #0x0                   	// #0
    17b8:	94000a12 	bl	4000 <sptNewTimer>
	sptStartTimer(timer);
    17bc:	528000b7 	mov	w23, #0x5                   	// #5
    17c0:	f94043e0 	ldr	x0, [sp, #128]
    17c4:	94000a2f 	bl	4080 <sptStartTimer>

	for(int it=0; it<niters; ++it) {
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    17c8:	0f000400 	movi	v0.2s, #0x0
    17cc:	f9400340 	ldr	x0, [x26]
    17d0:	94000615 	bl	3024 <sptConstantMatrix>
    17d4:	35000b00 	cbnz	w0, 1934 <main+0x4e4>
		if(dev_id == -2) {
    17d8:	b9407be0 	ldr	w0, [sp, #120]
    17dc:	3100081f 	cmn	w0, #0x2
    17e0:	540009e0 	b.eq	191c <main+0x4cc>  // b.none
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
		} else if(dev_id == -1) {
    17e4:	3100041f 	cmn	w0, #0x1
    17e8:	54000aa0 	b.eq	193c <main+0x4ec>  // b.none
	for(int it=0; it<niters; ++it) {
    17ec:	710006f7 	subs	w23, w23, #0x1
    17f0:	54fffec1 	b.ne	17c8 <main+0x378>  // b.any
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
#endif
		}
	}

	sptStopTimer(timer);
    17f4:	f94043e0 	ldr	x0, [sp, #128]
    17f8:	94000a52 	bl	4140 <sptStopTimer>

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    17fc:	f94043e0 	ldr	x0, [sp, #128]
    1800:	f0000002 	adrp	x2, 4000 <sptNewTimer>
    1804:	528000a1 	mov	w1, #0x5                   	// #5
    1808:	912c6042 	add	x2, x2, #0xb18
    180c:	94000aaa 	bl	42b4 <sptPrintAverageElapsedTime>
    1810:	1e604005 	fmov	d5, d0
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1814:	b94077e3 	ldr	w3, [sp, #116]
    1818:	1e630264 	ucvtf	d4, w19
    181c:	f94053e2 	ldr	x2, [sp, #160]
    1820:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1824:	1e630062 	ucvtf	d2, w3
    1828:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    182c:	9e630040 	ucvtf	d0, x2
    1830:	9e670001 	fmov	d1, x0
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1834:	9b020b62 	madd	x2, x27, x2, x2
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1838:	1e620884 	fmul	d4, d4, d2
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    183c:	d37ef442 	lsl	x2, x2, #2
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1840:	1e600884 	fmul	d4, d4, d0
    1844:	1e651884 	fdiv	d4, d4, d5
    1848:	1e611880 	fdiv	d0, d4, d1
	for (sptIndex m=0; m<nmodes; ++m) {
    184c:	340010d3 	cbz	w19, 1a64 <main+0x614>
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1850:	f9404fe4 	ldr	x4, [sp, #152]
    1854:	d2800000 	mov	x0, #0x0                   	// #0
    1858:	b8607881 	ldr	w1, [x4, x0, lsl #2]
	for (sptIndex m=0; m<nmodes; ++m) {
    185c:	91000400 	add	x0, x0, #0x1
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1860:	1b017c61 	mul	w1, w3, w1
    1864:	8b214842 	add	x2, x2, w1, uxtw #2
	for (sptIndex m=0; m<nmodes; ++m) {
    1868:	6b00027f 	cmp	w19, w0
    186c:	54ffff68 	b.hi	1858 <main+0x408>  // b.pmore
	}
	double gbw = (double)bytes / aver_time / 1e9;
    1870:	9e630042 	ucvtf	d2, x2
    1874:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1878:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    187c:	9e670001 	fmov	d1, x0
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1880:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1884:	912fa000 	add	x0, x0, #0xbe8
	double gbw = (double)bytes / aver_time / 1e9;
    1888:	1e651842 	fdiv	d2, d2, d5
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    188c:	1e611841 	fdiv	d1, d2, d1
    1890:	97fffedc 	bl	1400 <printf@plt>

	if(fo != NULL) {
    1894:	b4000116 	cbz	x22, 18b4 <main+0x464>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1898:	f94037e0 	ldr	x0, [sp, #104]
    189c:	aa1603e1 	mov	x1, x22
    18a0:	f8606a80 	ldr	x0, [x20, x0]
    18a4:	94000abf 	bl	43a0 <sptDumpMatrix>
    18a8:	35000460 	cbnz	w0, 1934 <main+0x4e4>
		fclose(fo);
    18ac:	aa1603e0 	mov	x0, x22
    18b0:	97fffe60 	bl	1230 <fclose@plt>
	}

	sptFreeTimer(timer);
    18b4:	f94043e0 	ldr	x0, [sp, #128]
		bytes += X.ndims[m] * R * sizeof(sptValue);
    18b8:	d2800016 	mov	x22, #0x0                   	// #0
	sptFreeTimer(timer);
    18bc:	94000a9d 	bl	4330 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
		sptFreeMatrix(U[m]);
    18c0:	f8767a80 	ldr	x0, [x20, x22, lsl #3]
	for(sptIndex m=0; m<nmodes; ++m) {
    18c4:	910006d6 	add	x22, x22, #0x1
		sptFreeMatrix(U[m]);
    18c8:	940005eb 	bl	3074 <sptFreeMatrix>
	for(sptIndex m=0; m<nmodes; ++m) {
    18cc:	6b16027f 	cmp	w19, w22
    18d0:	54ffff88 	b.hi	18c0 <main+0x470>  // b.pmore
	}
	sptFreeSparseTensor(&X);
    18d4:	aa1903e0 	mov	x0, x25
    18d8:	9400016b 	bl	1e84 <sptFreeSparseTensor>
	free(mats_order);
    18dc:	aa1803e0 	mov	x0, x24
    18e0:	97fffeb0 	bl	13a0 <free@plt>
	sptFreeMatrix(U[nmodes]);
    18e4:	f87b7a80 	ldr	x0, [x20, x27, lsl #3]
    18e8:	940005e3 	bl	3074 <sptFreeMatrix>
	free(U);
    18ec:	aa1403e0 	mov	x0, x20
    18f0:	97fffeac 	bl	13a0 <free@plt>

	if (!random){
    18f4:	34000355 	cbz	w21, 195c <main+0x50c>
		if(fPtr1 != NULL) fclose(fPtr1);
		if(fPtr2 != NULL) fclose(fPtr2);

	}
	return 0;
}
    18f8:	52800000 	mov	w0, #0x0                   	// #0
    18fc:	a9407bfd 	ldp	x29, x30, [sp]
    1900:	a94153f3 	ldp	x19, x20, [sp, #16]
    1904:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1908:	a94363f7 	ldp	x23, x24, [sp, #48]
    190c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    1910:	a94573fb 	ldp	x27, x28, [sp, #80]
    1914:	913203ff 	add	sp, sp, #0xc80
    1918:	d65f03c0 	ret
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    191c:	b94073e3 	ldr	w3, [sp, #112]
    1920:	aa1803e2 	mov	x2, x24
    1924:	aa1403e1 	mov	x1, x20
    1928:	aa1903e0 	mov	x0, x25
    192c:	940006f6 	bl	3504 <sptMTTKRP>
    1930:	34fff5e0 	cbz	w0, 17ec <main+0x39c>
				sptAssert(fo != NULL);
    1934:	12800000 	mov	w0, #0xffffffff            	// #-1
    1938:	97fffe22 	bl	11c0 <exit@plt>
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    193c:	b94073e3 	ldr	w3, [sp, #112]
    1940:	2a1c03e4 	mov	w4, w28
    1944:	aa1803e2 	mov	x2, x24
    1948:	aa1403e1 	mov	x1, x20
    194c:	aa1903e0 	mov	x0, x25
    1950:	94000931 	bl	3e14 <sptOmpMTTKRP>
    1954:	34fff4c0 	cbz	w0, 17ec <main+0x39c>
    1958:	17fffff7 	b	1934 <main+0x4e4>
		FILE* fPtr1 = fopen(fvname, "r");
    195c:	9112c3f6 	add	x22, sp, #0x4b0
    1960:	f0000013 	adrp	x19, 4000 <sptNewTimer>
    1964:	912cc273 	add	x19, x19, #0xb30
    1968:	aa1603e0 	mov	x0, x22
    196c:	aa1303e1 	mov	x1, x19
		FILE* fPtr2 = fopen(foname, "r");
    1970:	912263f5 	add	x21, sp, #0x898
		FILE* fPtr1 = fopen(fvname, "r");
    1974:	97fffe33 	bl	1240 <fopen@plt>
    1978:	aa0003f4 	mov	x20, x0
		FILE* fPtr2 = fopen(foname, "r");
    197c:	aa1303e1 	mov	x1, x19
    1980:	aa1503e0 	mov	x0, x21
    1984:	97fffe2f 	bl	1240 <fopen@plt>
    1988:	aa0003f3 	mov	x19, x0
		if (fPtr1 == NULL || fPtr2 == NULL) {
    198c:	f100029f 	cmp	x20, #0x0
    1990:	fa401804 	ccmp	x0, #0x0, #0x4, ne  // ne = any
    1994:	54000200 	b.eq	19d4 <main+0x584>  // b.none
			int diff = compareFile(fPtr1, fPtr2);
    1998:	aa0003e1 	mov	x1, x0
    199c:	aa1403e0 	mov	x0, x20
    19a0:	940000c4 	bl	1cb0 <compareFile>
			if (diff == 0) {
    19a4:	35000820 	cbnz	w0, 1aa8 <main+0x658>
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
    19a8:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    19ac:	aa1603e2 	mov	x2, x22
    19b0:	aa1503e1 	mov	x1, x21
    19b4:	912e4000 	add	x0, x0, #0xb90
    19b8:	97fffe92 	bl	1400 <printf@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    19bc:	aa1403e0 	mov	x0, x20
    19c0:	97fffe1c 	bl	1230 <fclose@plt>
		if(fPtr2 != NULL) fclose(fPtr2);
    19c4:	b4fff9b3 	cbz	x19, 18f8 <main+0x4a8>
    19c8:	aa1303e0 	mov	x0, x19
    19cc:	97fffe19 	bl	1230 <fclose@plt>
	return 0;
    19d0:	17ffffca 	b	18f8 <main+0x4a8>
			printf("\nUnable to open file.\n");
    19d4:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    19d8:	912ce000 	add	x0, x0, #0xb38
    19dc:	97fffe61 	bl	1360 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    19e0:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    19e4:	912d4000 	add	x0, x0, #0xb50
    19e8:	97fffe5e 	bl	1360 <puts@plt>
		if(fPtr1 != NULL) fclose(fPtr1);
    19ec:	b4fffed4 	cbz	x20, 19c4 <main+0x574>
    19f0:	17fffff3 	b	19bc <main+0x56c>
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    19f4:	aa1803e2 	mov	x2, x24
    19f8:	aa1403e1 	mov	x1, x20
    19fc:	aa1903e0 	mov	x0, x25
    1a00:	940006c1 	bl	3504 <sptMTTKRP>
    1a04:	34ffed20 	cbz	w0, 17a8 <main+0x358>
    1a08:	17ffffcb 	b	1934 <main+0x4e4>
		#pragma omp parallel
    1a0c:	52800024 	mov	w4, #0x1                   	// #1
    1a10:	910203f7 	add	x23, sp, #0x80
    1a14:	52800003 	mov	w3, #0x0                   	// #0
    1a18:	52800002 	mov	w2, #0x0                   	// #0
    1a1c:	aa1703e1 	mov	x1, x23
    1a20:	90000000 	adrp	x0, 1000 <g_last_error+0x1000>
    1a24:	91304000 	add	x0, x0, #0xc10
    1a28:	b90083e4 	str	w4, [sp, #128]
    1a2c:	97fffe7d 	bl	1420 <GOMP_parallel@plt>
    1a30:	b94083fc 	ldr	w28, [sp, #128]
        printf("\nnthreads: %d\n", nthreads);
    1a34:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1a38:	912c2000 	add	x0, x0, #0xb08
    1a3c:	2a1c03e1 	mov	w1, w28
    1a40:	97fffe70 	bl	1400 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1a44:	b94073e3 	ldr	w3, [sp, #112]
    1a48:	2a1c03e4 	mov	w4, w28
    1a4c:	aa1803e2 	mov	x2, x24
    1a50:	aa1403e1 	mov	x1, x20
    1a54:	aa1903e0 	mov	x0, x25
    1a58:	940008ef 	bl	3e14 <sptOmpMTTKRP>
    1a5c:	34ffeaa0 	cbz	w0, 17b0 <main+0x360>
    1a60:	17ffffb5 	b	1934 <main+0x4e4>
	double gbw = (double)bytes / aver_time / 1e9;
    1a64:	9e630043 	ucvtf	d3, x2
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1a68:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1a6c:	912fa000 	add	x0, x0, #0xbe8
	double gbw = (double)bytes / aver_time / 1e9;
    1a70:	1e651863 	fdiv	d3, d3, d5
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1a74:	1e611861 	fdiv	d1, d3, d1
    1a78:	97fffe62 	bl	1400 <printf@plt>
	if(fo != NULL) {
    1a7c:	b4000116 	cbz	x22, 1a9c <main+0x64c>
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1a80:	f94037e0 	ldr	x0, [sp, #104]
    1a84:	aa1603e1 	mov	x1, x22
    1a88:	f8606a80 	ldr	x0, [x20, x0]
    1a8c:	94000a45 	bl	43a0 <sptDumpMatrix>
    1a90:	35fff520 	cbnz	w0, 1934 <main+0x4e4>
		fclose(fo);
    1a94:	aa1603e0 	mov	x0, x22
    1a98:	97fffde6 	bl	1230 <fclose@plt>
	sptFreeTimer(timer);
    1a9c:	f94043e0 	ldr	x0, [sp, #128]
    1aa0:	94000a24 	bl	4330 <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1aa4:	17ffff8c 	b	18d4 <main+0x484>
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1aa8:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1aac:	912ee000 	add	x0, x0, #0xbb8
    1ab0:	97fffe2c 	bl	1360 <puts@plt>
    1ab4:	17ffffc2 	b	19bc <main+0x56c>
				print_usage(argv);
    1ab8:	f9400280 	ldr	x0, [x20]
    1abc:	9400005e 	bl	1c34 <print_usage.isra.0>
				exit(1);
    1ac0:	52800020 	mov	w0, #0x1                   	// #1
    1ac4:	97fffdbf 	bl	11c0 <exit@plt>
    1ac8:	d503201f 	nop
    1acc:	d503201f 	nop

0000000000001ad0 <init_have_lse_atomics>:
    1ad0:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1ad4:	d2800200 	mov	x0, #0x10                  	// #16
    1ad8:	910003fd 	mov	x29, sp
    1adc:	97fffe19 	bl	1340 <__getauxval@plt>
    1ae0:	53082000 	ubfx	w0, w0, #8, #1
    1ae4:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1ae8:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1aec:	390aa420 	strb	w0, [x1, #681]
    1af0:	d65f03c0 	ret

0000000000001af4 <_start>:
    1af4:	d280001d 	mov	x29, #0x0                   	// #0
    1af8:	d280001e 	mov	x30, #0x0                   	// #0
    1afc:	aa0003e5 	mov	x5, x0
    1b00:	f94003e1 	ldr	x1, [sp]
    1b04:	910023e2 	add	x2, sp, #0x8
    1b08:	910003e6 	mov	x6, sp
    1b0c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    1b10:	f947ec00 	ldr	x0, [x0, #4056]
    1b14:	b00000a3 	adrp	x3, 16000 <__FRAME_END__+0x10088>
    1b18:	f947e863 	ldr	x3, [x3, #4048]
    1b1c:	b00000a4 	adrp	x4, 16000 <__FRAME_END__+0x10088>
    1b20:	f947cc84 	ldr	x4, [x4, #3992]
    1b24:	97fffdd7 	bl	1280 <__libc_start_main@plt>
    1b28:	97fffe0a 	bl	1350 <abort@plt>

0000000000001b2c <call_weak_fn>:
    1b2c:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    1b30:	f947e400 	ldr	x0, [x0, #4040]
    1b34:	b4000040 	cbz	x0, 1b3c <call_weak_fn+0x10>
    1b38:	17fffdfe 	b	1330 <__gmon_start__@plt>
    1b3c:	d65f03c0 	ret

0000000000001b40 <deregister_tm_clones>:
    1b40:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1b44:	910aa000 	add	x0, x0, #0x2a8
    1b48:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1b4c:	910aa021 	add	x1, x1, #0x2a8
    1b50:	eb00003f 	cmp	x1, x0
    1b54:	540000c0 	b.eq	1b6c <deregister_tm_clones+0x2c>  // b.none
    1b58:	b00000a1 	adrp	x1, 16000 <__FRAME_END__+0x10088>
    1b5c:	f947d021 	ldr	x1, [x1, #4000]
    1b60:	b4000061 	cbz	x1, 1b6c <deregister_tm_clones+0x2c>
    1b64:	aa0103f0 	mov	x16, x1
    1b68:	d61f0200 	br	x16
    1b6c:	d65f03c0 	ret

0000000000001b70 <register_tm_clones>:
    1b70:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1b74:	910aa000 	add	x0, x0, #0x2a8
    1b78:	d00000a1 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    1b7c:	910aa021 	add	x1, x1, #0x2a8
    1b80:	cb000021 	sub	x1, x1, x0
    1b84:	d37ffc22 	lsr	x2, x1, #63
    1b88:	8b810c41 	add	x1, x2, x1, asr #3
    1b8c:	9341fc21 	asr	x1, x1, #1
    1b90:	b40000c1 	cbz	x1, 1ba8 <register_tm_clones+0x38>
    1b94:	b00000a2 	adrp	x2, 16000 <__FRAME_END__+0x10088>
    1b98:	f947f042 	ldr	x2, [x2, #4064]
    1b9c:	b4000062 	cbz	x2, 1ba8 <register_tm_clones+0x38>
    1ba0:	aa0203f0 	mov	x16, x2
    1ba4:	d61f0200 	br	x16
    1ba8:	d65f03c0 	ret
    1bac:	d503201f 	nop

0000000000001bb0 <__do_global_dtors_aux>:
    1bb0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1bb4:	910003fd 	mov	x29, sp
    1bb8:	f9000bf3 	str	x19, [sp, #16]
    1bbc:	d00000b3 	adrp	x19, 17000 <memcpy@GLIBC_2.17>
    1bc0:	394aa260 	ldrb	w0, [x19, #680]
    1bc4:	35000140 	cbnz	w0, 1bec <__do_global_dtors_aux+0x3c>
    1bc8:	b00000a0 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    1bcc:	f947d400 	ldr	x0, [x0, #4008]
    1bd0:	b4000080 	cbz	x0, 1be0 <__do_global_dtors_aux+0x30>
    1bd4:	d00000a0 	adrp	x0, 17000 <memcpy@GLIBC_2.17>
    1bd8:	f940ac00 	ldr	x0, [x0, #344]
    1bdc:	97fffd7d 	bl	11d0 <__cxa_finalize@plt>
    1be0:	97ffffd8 	bl	1b40 <deregister_tm_clones>
    1be4:	52800020 	mov	w0, #0x1                   	// #1
    1be8:	390aa260 	strb	w0, [x19, #680]
    1bec:	f9400bf3 	ldr	x19, [sp, #16]
    1bf0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1bf4:	d65f03c0 	ret
    1bf8:	d503201f 	nop
    1bfc:	d503201f 	nop

0000000000001c00 <frame_dummy>:
    1c00:	17ffffdc 	b	1b70 <register_tm_clones>
    1c04:	d503201f 	nop
    1c08:	d503201f 	nop
    1c0c:	d503201f 	nop

0000000000001c10 <main._omp_fn.0>:
		#pragma omp parallel
    1c10:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1c14:	910003fd 	mov	x29, sp
    1c18:	f9000bf3 	str	x19, [sp, #16]
    1c1c:	aa0003f3 	mov	x19, x0
            nthreads = omp_get_num_threads();
    1c20:	97fffdb8 	bl	1300 <omp_get_num_threads@plt>
    1c24:	b9000260 	str	w0, [x19]
		#pragma omp parallel
    1c28:	f9400bf3 	ldr	x19, [sp, #16]
    1c2c:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1c30:	d65f03c0 	ret

0000000000001c34 <print_usage.isra.0>:
static void print_usage(char ** argv) {
    1c34:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
	printf("Usage: %s [options] \n\n", argv[0]);
    1c38:	aa0003e1 	mov	x1, x0
    1c3c:	f0000002 	adrp	x2, 4000 <sptNewTimer>
static void print_usage(char ** argv) {
    1c40:	910003fd 	mov	x29, sp
	printf("Usage: %s [options] \n\n", argv[0]);
    1c44:	9120c040 	add	x0, x2, #0x830
    1c48:	97fffdee 	bl	1400 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1c4c:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c50:	91212000 	add	x0, x0, #0x848
    1c54:	97fffdc3 	bl	1360 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    1c58:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c5c:	9121e000 	add	x0, x0, #0x878
    1c60:	97fffdc0 	bl	1360 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1c64:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c68:	9122c000 	add	x0, x0, #0x8b0
    1c6c:	97fffdbd 	bl	1360 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    1c70:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c74:	91246000 	add	x0, x0, #0x918
    1c78:	97fffdba 	bl	1360 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    1c7c:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c80:	9125a000 	add	x0, x0, #0x968
    1c84:	97fffdb7 	bl	1360 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    1c88:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c8c:	9126a000 	add	x0, x0, #0x9a8
    1c90:	97fffdb4 	bl	1360 <puts@plt>
	printf("         --help\n");
    1c94:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1c98:	9128e000 	add	x0, x0, #0xa38
    1c9c:	97fffdb1 	bl	1360 <puts@plt>
}
    1ca0:	a8c17bfd 	ldp	x29, x30, [sp], #16
	printf("\n");
    1ca4:	52800140 	mov	w0, #0xa                   	// #10
    1ca8:	17fffde2 	b	1430 <putchar@plt>
    1cac:	d503201f 	nop

0000000000001cb0 <compareFile>:



int compareFile(FILE * fPtr1, FILE * fPtr2)
{
    1cb0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1cb4:	910003fd 	mov	x29, sp
    1cb8:	a90153f3 	stp	x19, x20, [sp, #16]
    1cbc:	aa0103f4 	mov	x20, x1
    1cc0:	f90013f5 	str	x21, [sp, #32]
    1cc4:	aa0003f5 	mov	x21, x0
    1cc8:	14000006 	b	1ce0 <compareFile+0x30>
		ch1 = fgetc(fPtr1);
		ch2 = fgetc(fPtr2);
		if (ch1 != ch2) {
			return -1;
		}
	} while (ch1 != EOF && ch2 != EOF);
    1ccc:	3100067f 	cmn	w19, #0x1
    1cd0:	1a9f07e2 	cset	w2, ne  // ne = any
    1cd4:	7100005f 	cmp	w2, #0x0
    1cd8:	3a411a64 	ccmn	w19, #0x1, #0x4, ne  // ne = any
    1cdc:	540001a0 	b.eq	1d10 <compareFile+0x60>  // b.none
		ch1 = fgetc(fPtr1);
    1ce0:	aa1503e0 	mov	x0, x21
    1ce4:	97fffd6b 	bl	1290 <fgetc@plt>
    1ce8:	2a0003f3 	mov	w19, w0
		ch2 = fgetc(fPtr2);
    1cec:	aa1403e0 	mov	x0, x20
    1cf0:	97fffd68 	bl	1290 <fgetc@plt>
		if (ch1 != ch2) {
    1cf4:	6b00027f 	cmp	w19, w0
    1cf8:	54fffea0 	b.eq	1ccc <compareFile+0x1c>  // b.none
			return -1;
    1cfc:	12800000 	mov	w0, #0xffffffff            	// #-1
	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
		return 0;
	else
		return -1;
}
    1d00:	a94153f3 	ldp	x19, x20, [sp, #16]
    1d04:	f94013f5 	ldr	x21, [sp, #32]
    1d08:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1d0c:	d65f03c0 	ret
	if (ch1 == EOF && ch2 == EOF)
    1d10:	4b0203e0 	neg	w0, w2
}
    1d14:	a94153f3 	ldp	x19, x20, [sp, #16]
    1d18:	f94013f5 	ldr	x21, [sp, #32]
    1d1c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1d20:	d65f03c0 	ret
    1d24:	d503201f 	nop
    1d28:	d503201f 	nop
    1d2c:	d503201f 	nop

0000000000001d30 <sptNewSparseTensor>:
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1d30:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    1d34:	910003fd 	mov	x29, sp
    1d38:	a9025bf5 	stp	x21, x22, [sp, #32]
    1d3c:	aa0003f6 	mov	x22, x0
    1d40:	aa0203f5 	mov	x21, x2
    1d44:	a90153f3 	stp	x19, x20, [sp, #16]
    1d48:	2a0103f3 	mov	w19, w1
	}
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
//	spt_CheckOSError(!tsr->ndims, "SpTns New");
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
	tsr->nnz = 0;
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1d4c:	52800314 	mov	w20, #0x18                  	// #24
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    1d50:	f9001bf7 	str	x23, [sp, #48]
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1d54:	d37e7e77 	ubfiz	x23, x19, #2, #32
	tsr->nmodes = nmodes;
    1d58:	b90002c1 	str	w1, [x22]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1d5c:	9bb47c34 	umull	x20, w1, w20
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    1d60:	aa1703e0 	mov	x0, x23
    1d64:	97fffd3f 	bl	1260 <malloc@plt>
    1d68:	f90006c0 	str	x0, [x22, #8]
	for(i = 0; i < nmodes; ++i) {
    1d6c:	340005b3 	cbz	w19, 1e20 <sptNewSparseTensor+0xf0>
    1d70:	d2800003 	mov	x3, #0x0                   	// #0
    1d74:	d503201f 	nop
		tsr->sortorder[i] = i;
    1d78:	b8237803 	str	w3, [x0, x3, lsl #2]
	for(i = 0; i < nmodes; ++i) {
    1d7c:	91000463 	add	x3, x3, #0x1
    1d80:	6b03027f 	cmp	w19, w3
    1d84:	54ffffa8 	b.hi	1d78 <sptNewSparseTensor+0x48>  // b.pmore
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1d88:	aa1703e0 	mov	x0, x23
    1d8c:	97fffd35 	bl	1260 <malloc@plt>
    1d90:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1d94:	aa1503e1 	mov	x1, x21
    1d98:	aa1703e2 	mov	x2, x23
    1d9c:	52800315 	mov	w21, #0x18                  	// #24
    1da0:	97fffd04 	bl	11b0 <memcpy@plt>
	tsr->nnz = 0;
    1da4:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1da8:	aa1403e0 	mov	x0, x20
    1dac:	9bb57e75 	umull	x21, w19, w21
    1db0:	97fffd2c 	bl	1260 <malloc@plt>
    1db4:	d2800014 	mov	x20, #0x0                   	// #0
    1db8:	f90012c0 	str	x0, [x22, #32]
//	spt_CheckOSError(!tsr->inds, "SpTns New");
	for(i = 0; i < nmodes; ++i) {
    1dbc:	14000004 	b	1dcc <sptNewSparseTensor+0x9c>
    1dc0:	eb1402bf 	cmp	x21, x20
    1dc4:	54000420 	b.eq	1e48 <sptNewSparseTensor+0x118>  // b.none
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1dc8:	f94012c0 	ldr	x0, [x22, #32]
    1dcc:	8b140000 	add	x0, x0, x20
    1dd0:	d2800002 	mov	x2, #0x0                   	// #0
    1dd4:	d2800001 	mov	x1, #0x0                   	// #0
	for(i = 0; i < nmodes; ++i) {
    1dd8:	91006294 	add	x20, x20, #0x18
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    1ddc:	940000e6 	bl	2174 <sptNewIndexVector>
    1de0:	2a0003f3 	mov	w19, w0
		spt_CheckError(result, "SpTns New", NULL);
    1de4:	34fffee0 	cbz	w0, 1dc0 <sptNewSparseTensor+0x90>
    1de8:	f0000002 	adrp	x2, 4000 <sptNewTimer>
    1dec:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1df0:	9131c042 	add	x2, x2, #0xc70
    1df4:	91326000 	add	x0, x0, #0xc98
    1df8:	2a1303e1 	mov	w1, w19
    1dfc:	d2800004 	mov	x4, #0x0                   	// #0
    1e00:	52800643 	mov	w3, #0x32                  	// #50
    1e04:	940009cb 	bl	4530 <spt_ComplainError>
	}
	result = sptNewValueVector(&tsr->values, 0, 0);
	spt_CheckError(result, "SpTns New", NULL);
	return 0;
}
    1e08:	2a1303e0 	mov	w0, w19
    1e0c:	a94153f3 	ldp	x19, x20, [sp, #16]
    1e10:	a9425bf5 	ldp	x21, x22, [sp, #32]
    1e14:	f9401bf7 	ldr	x23, [sp, #48]
    1e18:	a8c47bfd 	ldp	x29, x30, [sp], #64
    1e1c:	d65f03c0 	ret
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    1e20:	aa1703e0 	mov	x0, x23
    1e24:	97fffd0f 	bl	1260 <malloc@plt>
    1e28:	f9000ac0 	str	x0, [x22, #16]
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    1e2c:	aa1703e2 	mov	x2, x23
    1e30:	aa1503e1 	mov	x1, x21
    1e34:	97fffcdf 	bl	11b0 <memcpy@plt>
	tsr->nnz = 0;
    1e38:	f9000edf 	str	xzr, [x22, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    1e3c:	aa1403e0 	mov	x0, x20
    1e40:	97fffd08 	bl	1260 <malloc@plt>
    1e44:	f90012c0 	str	x0, [x22, #32]
	result = sptNewValueVector(&tsr->values, 0, 0);
    1e48:	9100a2c0 	add	x0, x22, #0x28
    1e4c:	d2800002 	mov	x2, #0x0                   	// #0
    1e50:	d2800001 	mov	x1, #0x0                   	// #0
    1e54:	9400002b 	bl	1f00 <sptNewValueVector>
    1e58:	2a0003f3 	mov	w19, w0
	spt_CheckError(result, "SpTns New", NULL);
    1e5c:	34fffd60 	cbz	w0, 1e08 <sptNewSparseTensor+0xd8>
    1e60:	2a1303e1 	mov	w1, w19
    1e64:	f0000002 	adrp	x2, 4000 <sptNewTimer>
    1e68:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1e6c:	9131c042 	add	x2, x2, #0xc70
    1e70:	91326000 	add	x0, x0, #0xc98
    1e74:	d2800004 	mov	x4, #0x0                   	// #0
    1e78:	528006a3 	mov	w3, #0x35                  	// #53
    1e7c:	940009ad 	bl	4530 <spt_ComplainError>
    1e80:	17ffffe2 	b	1e08 <sptNewSparseTensor+0xd8>

0000000000001e84 <sptFreeSparseTensor>:

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    1e84:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1e88:	910003fd 	mov	x29, sp
    1e8c:	a90153f3 	stp	x19, x20, [sp, #16]
    1e90:	aa0003f4 	mov	x20, x0
	sptIndex i;
	for(i = 0; i < tsr->nmodes; ++i) {
    1e94:	b9400000 	ldr	w0, [x0]
    1e98:	34000180 	cbz	w0, 1ec8 <sptFreeSparseTensor+0x44>
    1e9c:	52800013 	mov	w19, #0x0                   	// #0
    1ea0:	f90013f5 	str	x21, [sp, #32]
		sptFreeIndexVector(&tsr->inds[i]);
    1ea4:	52800315 	mov	w21, #0x18                  	// #24
    1ea8:	f9401280 	ldr	x0, [x20, #32]
    1eac:	9bb50260 	umaddl	x0, w19, w21, x0
	for(i = 0; i < tsr->nmodes; ++i) {
    1eb0:	11000673 	add	w19, w19, #0x1
		sptFreeIndexVector(&tsr->inds[i]);
    1eb4:	9400013c 	bl	23a4 <sptFreeIndexVector>
	for(i = 0; i < tsr->nmodes; ++i) {
    1eb8:	b9400280 	ldr	w0, [x20]
    1ebc:	6b13001f 	cmp	w0, w19
    1ec0:	54ffff48 	b.hi	1ea8 <sptFreeSparseTensor+0x24>  // b.pmore
    1ec4:	f94013f5 	ldr	x21, [sp, #32]
	}
	free(tsr->sortorder);
    1ec8:	f9400680 	ldr	x0, [x20, #8]
    1ecc:	97fffd35 	bl	13a0 <free@plt>
	free(tsr->ndims);
    1ed0:	f9400a80 	ldr	x0, [x20, #16]
    1ed4:	97fffd33 	bl	13a0 <free@plt>
	free(tsr->inds);
    1ed8:	f9401280 	ldr	x0, [x20, #32]
    1edc:	97fffd31 	bl	13a0 <free@plt>
	sptFreeValueVector(&tsr->values);
    1ee0:	9100a280 	add	x0, x20, #0x28
    1ee4:	940000a0 	bl	2164 <sptFreeValueVector>
	tsr->nmodes = 0;
    1ee8:	b900029f 	str	wzr, [x20]
}
    1eec:	a94153f3 	ldp	x19, x20, [sp, #16]
    1ef0:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1ef4:	d65f03c0 	ret
    1ef8:	d503201f 	nop
    1efc:	d503201f 	nop

0000000000001f00 <sptNewValueVector>:
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    1f00:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1f04:	f100083f 	cmp	x1, #0x2
    1f08:	d2800043 	mov	x3, #0x2                   	// #2
    1f0c:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    1f10:	910003fd 	mov	x29, sp
    1f14:	eb02007f 	cmp	x3, x2
    1f18:	aa0103e4 	mov	x4, x1
    1f1c:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
    1f20:	a90153f3 	stp	x19, x20, [sp, #16]
    1f24:	aa0003f3 	mov	x19, x0
	if(cap < 2) {
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    1f28:	d2800021 	mov	x1, #0x1                   	// #1
	vec->cap = cap;
    1f2c:	a9000804 	stp	x4, x2, [x0]
	vec->data = malloc(cap * sizeof *vec->data);
    1f30:	d37ef440 	lsl	x0, x2, #2
    1f34:	97fffce3 	bl	12c0 <calloc@plt>
    1f38:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "ValVec New");
    1f3c:	b40000a0 	cbz	x0, 1f50 <sptNewValueVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    1f40:	52800000 	mov	w0, #0x0                   	// #0
}
    1f44:	a94153f3 	ldp	x19, x20, [sp, #16]
    1f48:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1f4c:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "ValVec New");
    1f50:	97fffd30 	bl	1410 <__errno_location@plt>
    1f54:	aa0003f3 	mov	x19, x0
	return 0;
    1f58:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "ValVec New");
    1f5c:	b9400274 	ldr	w20, [x19]
    1f60:	3140429f 	cmn	w20, #0x10, lsl #12
    1f64:	54ffff00 	b.eq	1f44 <sptNewValueVector+0x44>  // b.none
    1f68:	2a1403e0 	mov	w0, w20
    1f6c:	97fffce9 	bl	1310 <strerror@plt>
    1f70:	11404281 	add	w1, w20, #0x10, lsl #12
    1f74:	aa0003e4 	mov	x4, x0
    1f78:	f0000002 	adrp	x2, 4000 <sptNewTimer>
    1f7c:	f0000000 	adrp	x0, 4000 <sptNewTimer>
    1f80:	9132a042 	add	x2, x2, #0xca8
    1f84:	91334000 	add	x0, x0, #0xcd0
    1f88:	528005c3 	mov	w3, #0x2e                  	// #46
    1f8c:	94000969 	bl	4530 <spt_ComplainError>
    1f90:	b9400260 	ldr	w0, [x19]
    1f94:	11404000 	add	w0, w0, #0x10, lsl #12
    1f98:	17ffffeb 	b	1f44 <sptNewValueVector+0x44>
    1f9c:	d503201f 	nop

0000000000001fa0 <sptConstantValueVector>:
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
	for(sptNnzIndex i=0; i<vec->len; ++i)
    1fa0:	f9400001 	ldr	x1, [x0]
    1fa4:	b40000c1 	cbz	x1, 1fbc <sptConstantValueVector+0x1c>
    1fa8:	f9400800 	ldr	x0, [x0, #16]
    1fac:	8b010801 	add	x1, x0, x1, lsl #2
		vec->data[i] = val;
    1fb0:	bc004400 	str	s0, [x0], #4
	for(sptNnzIndex i=0; i<vec->len; ++i)
    1fb4:	eb01001f 	cmp	x0, x1
    1fb8:	54ffffc1 	b.ne	1fb0 <sptConstantValueVector+0x10>  // b.any
	return 0;
}
    1fbc:	52800000 	mov	w0, #0x0                   	// #0
    1fc0:	d65f03c0 	ret

0000000000001fc4 <sptAppendValueVector>:
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    1fc4:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    1fc8:	910003fd 	mov	x29, sp
    1fcc:	a90153f3 	stp	x19, x20, [sp, #16]
    1fd0:	aa0003f3 	mov	x19, x0
	if(vec->cap <= vec->len) {
    1fd4:	a9400001 	ldp	x1, x0, [x0]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    1fd8:	f9400a74 	ldr	x20, [x19, #16]
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    1fdc:	fd001fe8 	str	d8, [sp, #56]
    1fe0:	1e204008 	fmov	s8, s0
	if(vec->cap <= vec->len) {
    1fe4:	eb01001f 	cmp	x0, x1
    1fe8:	54000129 	b.ls	200c <sptAppendValueVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "ValVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    1fec:	bc217a88 	str	s8, [x20, x1, lsl #2]
	++vec->len;
	return 0;
    1ff0:	52800000 	mov	w0, #0x0                   	// #0
	++vec->len;
    1ff4:	91000421 	add	x1, x1, #0x1
    1ff8:	f9000261 	str	x1, [x19]
}
    1ffc:	a94153f3 	ldp	x19, x20, [sp, #16]
    2000:	fd401fe8 	ldr	d8, [sp, #56]
    2004:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2008:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    200c:	a9025bf5 	stp	x21, x22, [sp, #32]
    2010:	8b400415 	add	x21, x0, x0, lsr #1
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2014:	aa1403e0 	mov	x0, x20
    2018:	d37ef6a1 	lsl	x1, x21, #2
    201c:	97fffcad 	bl	12d0 <realloc@plt>
    2020:	aa0003f4 	mov	x20, x0
		spt_CheckOSError(!newdata, "ValVec Append");
    2024:	b40000a0 	cbz	x0, 2038 <sptAppendValueVector+0x74>
		vec->data = newdata;
    2028:	a900d275 	stp	x21, x20, [x19, #8]
	vec->data[vec->len] = value;
    202c:	f9400261 	ldr	x1, [x19]
		vec->data = newdata;
    2030:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2034:	17ffffee 	b	1fec <sptAppendValueVector+0x28>
		spt_CheckOSError(!newdata, "ValVec Append");
    2038:	f9001bf7 	str	x23, [sp, #48]
    203c:	97fffcf5 	bl	1410 <__errno_location@plt>
    2040:	b9400017 	ldr	w23, [x0]
    2044:	aa0003f6 	mov	x22, x0
    2048:	314042ff 	cmn	w23, #0x10, lsl #12
    204c:	54000061 	b.ne	2058 <sptAppendValueVector+0x94>  // b.any
    2050:	f9401bf7 	ldr	x23, [sp, #48]
    2054:	17fffff5 	b	2028 <sptAppendValueVector+0x64>
    2058:	2a1703e0 	mov	w0, w23
    205c:	97fffcad 	bl	1310 <strerror@plt>
    2060:	114042e1 	add	w1, w23, #0x10, lsl #12
    2064:	aa0003e4 	mov	x4, x0
    2068:	d0000002 	adrp	x2, 4000 <sptNewTimer>
    206c:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    2070:	9132a042 	add	x2, x2, #0xca8
    2074:	91338000 	add	x0, x0, #0xce0
    2078:	52800a43 	mov	w3, #0x52                  	// #82
    207c:	9400092d 	bl	4530 <spt_ComplainError>
    2080:	b94002c0 	ldr	w0, [x22]
    2084:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2088:	11404000 	add	w0, w0, #0x10, lsl #12
    208c:	f9401bf7 	ldr	x23, [sp, #48]
    2090:	17ffffdb 	b	1ffc <sptAppendValueVector+0x38>

0000000000002094 <sptResizeValueVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2094:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2098:	f100083f 	cmp	x1, #0x2
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    209c:	910003fd 	mov	x29, sp
    20a0:	a90153f3 	stp	x19, x20, [sp, #16]
    20a4:	aa0003f3 	mov	x19, x0
    20a8:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    20ac:	f9400400 	ldr	x0, [x0, #8]
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    20b0:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    20b4:	d2800055 	mov	x21, #0x2                   	// #2
    20b8:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    20bc:	eb15001f 	cmp	x0, x21
    20c0:	540000e1 	b.ne	20dc <sptResizeValueVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    20c4:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    20c8:	f9000261 	str	x1, [x19]
}
    20cc:	a94153f3 	ldp	x19, x20, [sp, #16]
    20d0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    20d4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    20d8:	d65f03c0 	ret
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    20dc:	f9400a60 	ldr	x0, [x19, #16]
    20e0:	d37ef6a1 	lsl	x1, x21, #2
    20e4:	97fffc7b 	bl	12d0 <realloc@plt>
    20e8:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "ValVec Resize");
    20ec:	b4000100 	cbz	x0, 210c <sptResizeValueVector+0x78>
		vec->cap = newcap;
    20f0:	a9005674 	stp	x20, x21, [x19]
	return 0;
    20f4:	52800000 	mov	w0, #0x0                   	// #0
		vec->data = newdata;
    20f8:	f9000a76 	str	x22, [x19, #16]
}
    20fc:	a94153f3 	ldp	x19, x20, [sp, #16]
    2100:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2104:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2108:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "ValVec Resize");
    210c:	a90363f7 	stp	x23, x24, [sp, #48]
    2110:	97fffcc0 	bl	1410 <__errno_location@plt>
    2114:	b9400018 	ldr	w24, [x0]
    2118:	aa0003f7 	mov	x23, x0
    211c:	3140431f 	cmn	w24, #0x10, lsl #12
    2120:	54000061 	b.ne	212c <sptResizeValueVector+0x98>  // b.any
    2124:	a94363f7 	ldp	x23, x24, [sp, #48]
    2128:	17fffff2 	b	20f0 <sptResizeValueVector+0x5c>
    212c:	2a1803e0 	mov	w0, w24
    2130:	97fffc78 	bl	1310 <strerror@plt>
    2134:	11404301 	add	w1, w24, #0x10, lsl #12
    2138:	aa0003e4 	mov	x4, x0
    213c:	d0000002 	adrp	x2, 4000 <sptNewTimer>
    2140:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    2144:	9132a042 	add	x2, x2, #0xca8
    2148:	9133c000 	add	x0, x0, #0xcf0
    214c:	52800d23 	mov	w3, #0x69                  	// #105
    2150:	940008f8 	bl	4530 <spt_ComplainError>
    2154:	b94002e0 	ldr	w0, [x23]
    2158:	a94363f7 	ldp	x23, x24, [sp, #48]
    215c:	11404000 	add	w0, w0, #0x10, lsl #12
    2160:	17ffffdb 	b	20cc <sptResizeValueVector+0x38>

0000000000002164 <sptFreeValueVector>:
 * Release the memory buffer a value vector is holding
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
    2164:	aa0003e1 	mov	x1, x0
	vec->len = 0;
	vec->cap = 0;
	free(vec->data);
    2168:	f9400800 	ldr	x0, [x0, #16]
	vec->cap = 0;
    216c:	a9007c3f 	stp	xzr, xzr, [x1]
	free(vec->data);
    2170:	17fffc8c 	b	13a0 <free@plt>

0000000000002174 <sptNewIndexVector>:
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    2174:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2178:	f100083f 	cmp	x1, #0x2
    217c:	d2800043 	mov	x3, #0x2                   	// #2
    2180:	9a832023 	csel	x3, x1, x3, cs  // cs = hs, nlast
    2184:	910003fd 	mov	x29, sp
    2188:	eb02007f 	cmp	x3, x2
    218c:	aa0103e4 	mov	x4, x1
    2190:	9a822062 	csel	x2, x3, x2, cs  // cs = hs, nlast
    2194:	a90153f3 	stp	x19, x20, [sp, #16]
    2198:	aa0003f3 	mov	x19, x0
	if(cap < 2) {
		cap = 2;
	}
	vec->len = len;
	vec->cap = cap;
	vec->data = malloc(cap * sizeof *vec->data);
    219c:	d2800021 	mov	x1, #0x1                   	// #1
	vec->cap = cap;
    21a0:	a9000804 	stp	x4, x2, [x0]
	vec->data = malloc(cap * sizeof *vec->data);
    21a4:	d37ef440 	lsl	x0, x2, #2
    21a8:	97fffc46 	bl	12c0 <calloc@plt>
    21ac:	f9000a60 	str	x0, [x19, #16]
	spt_CheckOSError(!vec->data, "IdxVec New");
    21b0:	b40000a0 	cbz	x0, 21c4 <sptNewIndexVector+0x50>
	memset(vec->data, 0, cap * sizeof *vec->data);
	return 0;
    21b4:	52800000 	mov	w0, #0x0                   	// #0
}
    21b8:	a94153f3 	ldp	x19, x20, [sp, #16]
    21bc:	a8c27bfd 	ldp	x29, x30, [sp], #32
    21c0:	d65f03c0 	ret
	spt_CheckOSError(!vec->data, "IdxVec New");
    21c4:	97fffc93 	bl	1410 <__errno_location@plt>
    21c8:	aa0003f3 	mov	x19, x0
	return 0;
    21cc:	52800000 	mov	w0, #0x0                   	// #0
	spt_CheckOSError(!vec->data, "IdxVec New");
    21d0:	b9400274 	ldr	w20, [x19]
    21d4:	3140429f 	cmn	w20, #0x10, lsl #12
    21d8:	54ffff00 	b.eq	21b8 <sptNewIndexVector+0x44>  // b.none
    21dc:	2a1403e0 	mov	w0, w20
    21e0:	97fffc4c 	bl	1310 <strerror@plt>
    21e4:	11404281 	add	w1, w20, #0x10, lsl #12
    21e8:	aa0003e4 	mov	x4, x0
    21ec:	d0000002 	adrp	x2, 4000 <sptNewTimer>
    21f0:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    21f4:	9132a042 	add	x2, x2, #0xca8
    21f8:	91340000 	add	x0, x0, #0xd00
    21fc:	52801283 	mov	w3, #0x94                  	// #148
    2200:	940008cc 	bl	4530 <spt_ComplainError>
    2204:	b9400260 	ldr	w0, [x19]
    2208:	11404000 	add	w0, w0, #0x10, lsl #12
    220c:	17ffffeb 	b	21b8 <sptNewIndexVector+0x44>

0000000000002210 <sptAppendIndexVector>:
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    2210:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2214:	910003fd 	mov	x29, sp
    2218:	a90153f3 	stp	x19, x20, [sp, #16]
    221c:	aa0003f3 	mov	x19, x0
    2220:	2a0103f4 	mov	w20, w1
	if(vec->cap <= vec->len) {
    2224:	a9400002 	ldp	x2, x0, [x0]
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    2228:	a9025bf5 	stp	x21, x22, [sp, #32]
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    222c:	f9400a75 	ldr	x21, [x19, #16]
	if(vec->cap <= vec->len) {
    2230:	eb02001f 	cmp	x0, x2
    2234:	54000129 	b.ls	2258 <sptAppendIndexVector+0x48>  // b.plast
		spt_CheckOSError(!newdata, "IdxVec Append");
		vec->cap = newcap;
		vec->data = newdata;
	}
	vec->data[vec->len] = value;
    2238:	b8227ab4 	str	w20, [x21, x2, lsl #2]
	++vec->len;
	return 0;
    223c:	52800000 	mov	w0, #0x0                   	// #0
	++vec->len;
    2240:	91000442 	add	x2, x2, #0x1
    2244:	f9000262 	str	x2, [x19]
}
    2248:	a94153f3 	ldp	x19, x20, [sp, #16]
    224c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2250:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2254:	d65f03c0 	ret
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    2258:	8b400416 	add	x22, x0, x0, lsr #1
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    225c:	aa1503e0 	mov	x0, x21
    2260:	d37ef6c1 	lsl	x1, x22, #2
    2264:	97fffc1b 	bl	12d0 <realloc@plt>
    2268:	aa0003f5 	mov	x21, x0
		spt_CheckOSError(!newdata, "IdxVec Append");
    226c:	b4000080 	cbz	x0, 227c <sptAppendIndexVector+0x6c>
		vec->data = newdata;
    2270:	a900d676 	stp	x22, x21, [x19, #8]
	vec->data[vec->len] = value;
    2274:	f9400262 	ldr	x2, [x19]
    2278:	17fffff0 	b	2238 <sptAppendIndexVector+0x28>
		spt_CheckOSError(!newdata, "IdxVec Append");
    227c:	a90363f7 	stp	x23, x24, [sp, #48]
    2280:	97fffc64 	bl	1410 <__errno_location@plt>
    2284:	b9400018 	ldr	w24, [x0]
    2288:	aa0003f7 	mov	x23, x0
    228c:	3140431f 	cmn	w24, #0x10, lsl #12
    2290:	54000061 	b.ne	229c <sptAppendIndexVector+0x8c>  // b.any
    2294:	a94363f7 	ldp	x23, x24, [sp, #48]
    2298:	17fffff6 	b	2270 <sptAppendIndexVector+0x60>
    229c:	2a1803e0 	mov	w0, w24
    22a0:	97fffc1c 	bl	1310 <strerror@plt>
    22a4:	11404301 	add	w1, w24, #0x10, lsl #12
    22a8:	aa0003e4 	mov	x4, x0
    22ac:	d0000002 	adrp	x2, 4000 <sptNewTimer>
    22b0:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    22b4:	9132a042 	add	x2, x2, #0xca8
    22b8:	91344000 	add	x0, x0, #0xd10
    22bc:	52801543 	mov	w3, #0xaa                  	// #170
    22c0:	9400089c 	bl	4530 <spt_ComplainError>
    22c4:	b94002e0 	ldr	w0, [x23]
    22c8:	a94363f7 	ldp	x23, x24, [sp, #48]
    22cc:	11404000 	add	w0, w0, #0x10, lsl #12
    22d0:	17ffffde 	b	2248 <sptAppendIndexVector+0x38>

00000000000022d4 <sptResizeIndexVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    22d4:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
	sptNnzIndex newcap = size < 2 ? 2 : size;
    22d8:	f100083f 	cmp	x1, #0x2
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    22dc:	910003fd 	mov	x29, sp
    22e0:	a90153f3 	stp	x19, x20, [sp, #16]
    22e4:	aa0003f3 	mov	x19, x0
    22e8:	aa0103f4 	mov	x20, x1
	if(newcap != vec->cap) {
    22ec:	f9400400 	ldr	x0, [x0, #8]
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    22f0:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    22f4:	d2800055 	mov	x21, #0x2                   	// #2
    22f8:	9a952035 	csel	x21, x1, x21, cs  // cs = hs, nlast
	if(newcap != vec->cap) {
    22fc:	eb15001f 	cmp	x0, x21
    2300:	540000e1 	b.ne	231c <sptResizeIndexVector+0x48>  // b.any
		vec->cap = newcap;
		vec->data = newdata;
	} else {
		vec->len = size;
	}
	return 0;
    2304:	52800000 	mov	w0, #0x0                   	// #0
		vec->len = size;
    2308:	f9000261 	str	x1, [x19]
}
    230c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2310:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2314:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2318:	d65f03c0 	ret
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    231c:	f9400a60 	ldr	x0, [x19, #16]
    2320:	d37ef6a1 	lsl	x1, x21, #2
    2324:	97fffbeb 	bl	12d0 <realloc@plt>
    2328:	aa0003f6 	mov	x22, x0
		spt_CheckOSError(!newdata, "IdxVec Resize");
    232c:	b4000100 	cbz	x0, 234c <sptResizeIndexVector+0x78>
		vec->cap = newcap;
    2330:	a9005674 	stp	x20, x21, [x19]
	return 0;
    2334:	52800000 	mov	w0, #0x0                   	// #0
		vec->data = newdata;
    2338:	f9000a76 	str	x22, [x19, #16]
}
    233c:	a94153f3 	ldp	x19, x20, [sp, #16]
    2340:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2344:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2348:	d65f03c0 	ret
		spt_CheckOSError(!newdata, "IdxVec Resize");
    234c:	a90363f7 	stp	x23, x24, [sp, #48]
    2350:	97fffc30 	bl	1410 <__errno_location@plt>
    2354:	b9400018 	ldr	w24, [x0]
    2358:	aa0003f7 	mov	x23, x0
    235c:	3140431f 	cmn	w24, #0x10, lsl #12
    2360:	54000061 	b.ne	236c <sptResizeIndexVector+0x98>  // b.any
    2364:	a94363f7 	ldp	x23, x24, [sp, #48]
    2368:	17fffff2 	b	2330 <sptResizeIndexVector+0x5c>
    236c:	2a1803e0 	mov	w0, w24
    2370:	97fffbe8 	bl	1310 <strerror@plt>
    2374:	11404301 	add	w1, w24, #0x10, lsl #12
    2378:	aa0003e4 	mov	x4, x0
    237c:	d0000002 	adrp	x2, 4000 <sptNewTimer>
    2380:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    2384:	9132a042 	add	x2, x2, #0xca8
    2388:	91348000 	add	x0, x0, #0xd20
    238c:	52801823 	mov	w3, #0xc1                  	// #193
    2390:	94000868 	bl	4530 <spt_ComplainError>
    2394:	b94002e0 	ldr	w0, [x23]
    2398:	a94363f7 	ldp	x23, x24, [sp, #48]
    239c:	11404000 	add	w0, w0, #0x10, lsl #12
    23a0:	17ffffdb 	b	230c <sptResizeIndexVector+0x38>

00000000000023a4 <sptFreeIndexVector>:
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    23a4:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    23a8:	910003fd 	mov	x29, sp
    23ac:	f9000bf3 	str	x19, [sp, #16]
    23b0:	aa0003f3 	mov	x19, x0
	free(vec->data);
    23b4:	f9400800 	ldr	x0, [x0, #16]
    23b8:	97fffbfa 	bl	13a0 <free@plt>
	vec->len = 0;
	vec->cap = 0;
    23bc:	a9007e7f 	stp	xzr, xzr, [x19]
}
    23c0:	f9400bf3 	ldr	x19, [sp, #16]
    23c4:	a8c27bfd 	ldp	x29, x30, [sp], #32
    23c8:	d65f03c0 	ret
    23cc:	d503201f 	nop

00000000000023d0 <fill_binary_nnzidx._omp_fn.0>:
		sptIndex const BUF_LEN = 1024*1024;
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    23d0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    23d4:	910003fd 	mov	x29, sp
    23d8:	f90013f5 	str	x21, [sp, #32]
    23dc:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    23e0:	34000355 	cbz	w21, 2448 <fill_binary_nnzidx._omp_fn.0+0x78>
    23e4:	a90153f3 	stp	x19, x20, [sp, #16]
    23e8:	aa0003f4 	mov	x20, x0
    23ec:	97fffbc5 	bl	1300 <omp_get_num_threads@plt>
    23f0:	2a0003f3 	mov	w19, w0
    23f4:	97fffb8b 	bl	1220 <omp_get_thread_num@plt>
    23f8:	1ad30aa2 	udiv	w2, w21, w19
    23fc:	1b13d441 	msub	w1, w2, w19, w21
    2400:	6b01001f 	cmp	w0, w1
    2404:	54000283 	b.cc	2454 <fill_binary_nnzidx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2408:	1b000441 	madd	w1, w2, w0, w1
    240c:	0b010043 	add	w3, w2, w1
    2410:	6b03003f 	cmp	w1, w3
    2414:	54000182 	b.cs	2444 <fill_binary_nnzidx._omp_fn.0+0x74>  // b.hs, b.nlast
    2418:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    241c:	b9401284 	ldr	w4, [x20, #16]
    2420:	0b010080 	add	w0, w4, w1
    2424:	0b030083 	add	w3, w4, w3
    2428:	8b214842 	add	x2, x2, w1, uxtw #2
    242c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2430:	b8404441 	ldr	w1, [x2], #4
    2434:	f82058a1 	str	x1, [x5, w0, uxtw #3]
    2438:	11000400 	add	w0, w0, #0x1
    243c:	6b00007f 	cmp	w3, w0
    2440:	54ffff81 	b.ne	2430 <fill_binary_nnzidx._omp_fn.0+0x60>  // b.any
    2444:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    2448:	f94013f5 	ldr	x21, [sp, #32]
    244c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2450:	d65f03c0 	ret
    2454:	11000442 	add	w2, w2, #0x1
    2458:	52800001 	mov	w1, #0x0                   	// #0
    245c:	17ffffeb 	b	2408 <fill_binary_nnzidx._omp_fn.0+0x38>

0000000000002460 <fill_binary_idx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    2460:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2464:	910003fd 	mov	x29, sp
    2468:	f90013f5 	str	x21, [sp, #32]
    246c:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2470:	34000355 	cbz	w21, 24d8 <fill_binary_idx._omp_fn.0+0x78>
    2474:	a90153f3 	stp	x19, x20, [sp, #16]
    2478:	aa0003f4 	mov	x20, x0
    247c:	97fffba1 	bl	1300 <omp_get_num_threads@plt>
    2480:	2a0003f3 	mov	w19, w0
    2484:	97fffb67 	bl	1220 <omp_get_thread_num@plt>
    2488:	1ad30aa2 	udiv	w2, w21, w19
    248c:	1b13d441 	msub	w1, w2, w19, w21
    2490:	6b01001f 	cmp	w0, w1
    2494:	54000283 	b.cc	24e4 <fill_binary_idx._omp_fn.0+0x84>  // b.lo, b.ul, b.last
    2498:	1b000441 	madd	w1, w2, w0, w1
    249c:	0b010043 	add	w3, w2, w1
    24a0:	6b03003f 	cmp	w1, w3
    24a4:	54000182 	b.cs	24d4 <fill_binary_idx._omp_fn.0+0x74>  // b.hs, b.nlast
    24a8:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    24ac:	b9401284 	ldr	w4, [x20, #16]
    24b0:	0b010080 	add	w0, w4, w1
    24b4:	0b030083 	add	w3, w4, w3
    24b8:	8b214842 	add	x2, x2, w1, uxtw #2
    24bc:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    24c0:	b8404441 	ldr	w1, [x2], #4
    24c4:	b82058a1 	str	w1, [x5, w0, uxtw #2]
    24c8:	11000400 	add	w0, w0, #0x1
    24cc:	6b00007f 	cmp	w3, w0
    24d0:	54ffff81 	b.ne	24c0 <fill_binary_idx._omp_fn.0+0x60>  // b.any
    24d4:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    24d8:	f94013f5 	ldr	x21, [sp, #32]
    24dc:	a8c37bfd 	ldp	x29, x30, [sp], #48
    24e0:	d65f03c0 	ret
    24e4:	11000442 	add	w2, w2, #0x1
    24e8:	52800001 	mov	w1, #0x0                   	// #0
    24ec:	17ffffeb 	b	2498 <fill_binary_idx._omp_fn.0+0x38>

00000000000024f0 <fill_binary_val._omp_fn.0>:
#endif

		for(sptIndex n=0; n < count; n += BUF_LEN) {
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
			fread(ubuf, sizeof(*ubuf), read_count, fin);
#pragma omp parallel for schedule(static)
    24f0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    24f4:	910003fd 	mov	x29, sp
    24f8:	f90013f5 	str	x21, [sp, #32]
    24fc:	b9401415 	ldr	w21, [x0, #20]
			for(sptIndex i=0; i < read_count; ++i) {
    2500:	34000375 	cbz	w21, 256c <fill_binary_val._omp_fn.0+0x7c>
    2504:	a90153f3 	stp	x19, x20, [sp, #16]
    2508:	aa0003f4 	mov	x20, x0
    250c:	97fffb7d 	bl	1300 <omp_get_num_threads@plt>
    2510:	2a0003f3 	mov	w19, w0
    2514:	97fffb43 	bl	1220 <omp_get_thread_num@plt>
    2518:	1ad30aa2 	udiv	w2, w21, w19
    251c:	1b13d441 	msub	w1, w2, w19, w21
    2520:	6b01001f 	cmp	w0, w1
    2524:	540002a3 	b.cc	2578 <fill_binary_val._omp_fn.0+0x88>  // b.lo, b.ul, b.last
    2528:	1b000441 	madd	w1, w2, w0, w1
    252c:	0b010043 	add	w3, w2, w1
    2530:	6b03003f 	cmp	w1, w3
    2534:	540001a2 	b.cs	2568 <fill_binary_val._omp_fn.0+0x78>  // b.hs, b.nlast
    2538:	a9400a85 	ldp	x5, x2, [x20]
#pragma omp parallel for schedule(static)
    253c:	b9401284 	ldr	w4, [x20, #16]
    2540:	0b010080 	add	w0, w4, w1
    2544:	0b030083 	add	w3, w4, w3
    2548:	8b214c42 	add	x2, x2, w1, uxtw #3
    254c:	d503201f 	nop
				buffer[n + i] = ubuf[i];
    2550:	fc408440 	ldr	d0, [x2], #8
    2554:	1e624000 	fcvt	s0, d0
    2558:	bc2058a0 	str	s0, [x5, w0, uxtw #2]
    255c:	11000400 	add	w0, w0, #0x1
    2560:	6b00007f 	cmp	w3, w0
    2564:	54ffff61 	b.ne	2550 <fill_binary_val._omp_fn.0+0x60>  // b.any
    2568:	a94153f3 	ldp	x19, x20, [sp, #16]
#pragma omp parallel for schedule(static)
    256c:	f94013f5 	ldr	x21, [sp, #32]
    2570:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2574:	d65f03c0 	ret
    2578:	11000442 	add	w2, w2, #0x1
    257c:	52800001 	mov	w1, #0x0                   	// #0
    2580:	17ffffea 	b	2528 <fill_binary_val._omp_fn.0+0x38>

0000000000002584 <fill_binary_idx.isra.0>:
static void fill_binary_idx(
    2584:	a9b87bfd 	stp	x29, x30, [sp, #-128]!
    2588:	910003fd 	mov	x29, sp
    258c:	a9025bf5 	stp	x21, x22, [sp, #32]
    2590:	2a0103f5 	mov	w21, w1
	if(header->idx_width == sizeof(sptIndex)) {
    2594:	f100105f 	cmp	x2, #0x4
    2598:	54000540 	b.eq	2640 <fill_binary_idx.isra.0+0xbc>  // b.none
    259c:	a90363f7 	stp	x23, x24, [sp, #48]
    25a0:	aa0003f7 	mov	x23, x0
    25a4:	aa0303f8 	mov	x24, x3
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    25a8:	d2a00800 	mov	x0, #0x400000              	// #4194304
    25ac:	97fffb2d 	bl	1260 <malloc@plt>
    25b0:	aa0003f6 	mov	x22, x0
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    25b4:	340003d5 	cbz	w21, 262c <fill_binary_idx.isra.0+0xa8>
    25b8:	a9046bf9 	stp	x25, x26, [sp, #64]
    25bc:	90000019 	adrp	x25, 2000 <sptAppendValueVector+0x3c>
    25c0:	91118339 	add	x25, x25, #0x460
    25c4:	f9002bfb 	str	x27, [sp, #80]
    25c8:	9101a3fb 	add	x27, sp, #0x68
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    25cc:	52a0021a 	mov	w26, #0x100000              	// #1048576
    25d0:	a90153f3 	stp	x19, x20, [sp, #16]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    25d4:	52800014 	mov	w20, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    25d8:	4b1402b3 	sub	w19, w21, w20
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    25dc:	aa1803e3 	mov	x3, x24
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    25e0:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    25e4:	d2800081 	mov	x1, #0x4                   	// #4
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    25e8:	1a9a9273 	csel	w19, w19, w26, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    25ec:	aa1603e0 	mov	x0, x22
    25f0:	2a1303e2 	mov	w2, w19
    25f4:	97fffb67 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    25f8:	a906dbf7 	stp	x23, x22, [sp, #104]
    25fc:	aa1b03e1 	mov	x1, x27
    2600:	aa1903e0 	mov	x0, x25
    2604:	52800003 	mov	w3, #0x0                   	// #0
    2608:	52800002 	mov	w2, #0x0                   	// #0
    260c:	290f4ff4 	stp	w20, w19, [sp, #120]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2610:	11440294 	add	w20, w20, #0x100, lsl #12
    2614:	97fffb83 	bl	1420 <GOMP_parallel@plt>
    2618:	6b1402bf 	cmp	w21, w20
    261c:	54fffde8 	b.hi	25d8 <fill_binary_idx.isra.0+0x54>  // b.pmore
    2620:	a94153f3 	ldp	x19, x20, [sp, #16]
    2624:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2628:	f9402bfb 	ldr	x27, [sp, #80]
		free(ubuf);
    262c:	aa1603e0 	mov	x0, x22
}
    2630:	a9425bf5 	ldp	x21, x22, [sp, #32]
		free(ubuf);
    2634:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    2638:	a8c87bfd 	ldp	x29, x30, [sp], #128
		free(ubuf);
    263c:	17fffb59 	b	13a0 <free@plt>
}
    2640:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2644:	aa0203e4 	mov	x4, x2
    2648:	a8c87bfd 	ldp	x29, x30, [sp], #128
		fread(buffer, sizeof(sptIndex), count, fin);
    264c:	2a0103e2 	mov	w2, w1
    2650:	aa0403e1 	mov	x1, x4
    2654:	17fffb4f 	b	1390 <fread@plt>
    2658:	d503201f 	nop
    265c:	d503201f 	nop

0000000000002660 <sptLoadSparseTensor>:
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname)
{
    2660:	a9b57bfd 	stp	x29, x30, [sp, #-176]!
    2664:	910003fd 	mov	x29, sp
    2668:	a90153f3 	stp	x19, x20, [sp, #16]
    266c:	aa0003f3 	mov	x19, x0
	FILE * fp = fopen(fname, "r");
    2670:	aa0203e0 	mov	x0, x2
{
    2674:	a9025bf5 	stp	x21, x22, [sp, #32]
    2678:	2a0103f6 	mov	w22, w1
	FILE * fp = fopen(fname, "r");
    267c:	d0000001 	adrp	x1, 4000 <sptNewTimer>
    2680:	912cc021 	add	x1, x1, #0xb30
{
    2684:	a90363f7 	stp	x23, x24, [sp, #48]
    2688:	aa0203f7 	mov	x23, x2
	FILE * fp = fopen(fname, "r");
    268c:	97fffaed 	bl	1240 <fopen@plt>
	sptAssert(fp != NULL);
    2690:	a9046bf9 	stp	x25, x26, [sp, #64]
    2694:	b4003d60 	cbz	x0, 2e40 <sptLoadSparseTensor+0x7e0>
	char const * const suffix = strrchr(fname, '.');
    2698:	aa0003f5 	mov	x21, x0
    269c:	528005c1 	mov	w1, #0x2e                  	// #46
    26a0:	aa1703e0 	mov	x0, x23
    26a4:	97fffb1f 	bl	1320 <strrchr@plt>
    26a8:	aa0003fa 	mov	x26, x0
	if(suffix == NULL) {
    26ac:	b4000f00 	cbz	x0, 288c <sptLoadSparseTensor+0x22c>
    26b0:	d0000004 	adrp	x4, 4000 <sptNewTimer>
    26b4:	900000b8 	adrp	x24, 16000 <__FRAME_END__+0x10088>
    26b8:	9134c081 	add	x1, x4, #0xd30
	} while(file_extensions[++idx].extension != NULL);
    26bc:	91350319 	add	x25, x24, #0xd40
	size_t idx = 0;
    26c0:	d2800014 	mov	x20, #0x0                   	// #0
    26c4:	14000005 	b	26d8 <sptLoadSparseTensor+0x78>
	} while(file_extensions[++idx].extension != NULL);
    26c8:	91000694 	add	x20, x20, #0x1
    26cc:	d37cee83 	lsl	x3, x20, #4
    26d0:	f8796861 	ldr	x1, [x3, x25]
    26d4:	b4000dc1 	cbz	x1, 288c <sptLoadSparseTensor+0x22c>
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    26d8:	aa1a03e0 	mov	x0, x26
    26dc:	97fffb29 	bl	1380 <strcmp@plt>
    26e0:	35ffff40 	cbnz	w0, 26c8 <sptLoadSparseTensor+0x68>
			return file_extensions[idx].type;
    26e4:	91350318 	add	x24, x24, #0xd40
    26e8:	8b141314 	add	x20, x24, x20, lsl #4
    26ec:	b9400a80 	ldr	w0, [x20, #8]

	int iores;
	switch(get_file_type(fname)) {
    26f0:	34000dc0 	cbz	w0, 28a8 <sptLoadSparseTensor+0x248>
    26f4:	7100041f 	cmp	w0, #0x1
    26f8:	54000b81 	b.ne	2868 <sptLoadSparseTensor+0x208>  // b.any
	fread(&(header->magic), sizeof(header->magic), 1, fin);
    26fc:	aa1503e3 	mov	x3, x21
    2700:	d2800022 	mov	x2, #0x1                   	// #1
    2704:	d2800081 	mov	x1, #0x4                   	// #4
    2708:	910203e0 	add	x0, sp, #0x80
    270c:	97fffb21 	bl	1390 <fread@plt>
	fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
    2710:	aa1503e3 	mov	x3, x21
    2714:	d2800022 	mov	x2, #0x1                   	// #1
    2718:	d2800101 	mov	x1, #0x8                   	// #8
    271c:	910223e0 	add	x0, sp, #0x88
    2720:	97fffb1c 	bl	1390 <fread@plt>
	fread(&(header->val_width), sizeof(header->val_width), 1, fin);
    2724:	d2800022 	mov	x2, #0x1                   	// #1
    2728:	aa1503e3 	mov	x3, x21
    272c:	910243e0 	add	x0, sp, #0x90
    2730:	d2800101 	mov	x1, #0x8                   	// #8
    2734:	97fffb17 	bl	1390 <fread@plt>
	if(header->idx_width > PASTA_INDEX_TYPEWIDTH / 8) {
    2738:	f94047e2 	ldr	x2, [sp, #136]
    273c:	f100105f 	cmp	x2, #0x4
    2740:	54003868 	b.hi	2e4c <sptLoadSparseTensor+0x7ec>  // b.pmore
	if(header->val_width > PASTA_VALUE_TYPEWIDTH / 8) {
    2744:	f9404be0 	ldr	x0, [sp, #144]
    2748:	f100101f 	cmp	x0, #0x4
    274c:	54001f08 	b.hi	2b2c <sptLoadSparseTensor+0x4cc>  // b.pmore
	fill_binary_idx(&nmodes, 1, &header, fin);
    2750:	aa1503e3 	mov	x3, x21
    2754:	52800021 	mov	w1, #0x1                   	// #1
    2758:	9101d3e0 	add	x0, sp, #0x74
	sptIndex nmodes = 0;
    275c:	b90077ff 	str	wzr, [sp, #116]
	sptNnzIndex nnz = 0;
    2760:	f9003fff 	str	xzr, [sp, #120]
	fill_binary_idx(&nmodes, 1, &header, fin);
    2764:	97ffff88 	bl	2584 <fill_binary_idx.isra.0>
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    2768:	b94077f4 	ldr	w20, [sp, #116]
    276c:	d37e7e80 	ubfiz	x0, x20, #2, #32
    2770:	97fffabc 	bl	1260 <malloc@plt>
	fill_binary_idx(dims, nmodes, &header, fin);
    2774:	f94047e2 	ldr	x2, [sp, #136]
    2778:	2a1403e1 	mov	w1, w20
    277c:	aa1503e3 	mov	x3, x21
	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    2780:	aa0003f4 	mov	x20, x0
	fill_binary_idx(dims, nmodes, &header, fin);
    2784:	97ffff80 	bl	2584 <fill_binary_idx.isra.0>
	if(header->idx_width == sizeof(sptNnzIndex)) {
    2788:	f94047e1 	ldr	x1, [sp, #136]
    278c:	f100203f 	cmp	x1, #0x8
    2790:	54001c40 	b.eq	2b18 <sptLoadSparseTensor+0x4b8>  // b.none
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    2794:	d2a00800 	mov	x0, #0x400000              	// #4194304
    2798:	97fffab2 	bl	1260 <malloc@plt>
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    279c:	aa1503e3 	mov	x3, x21
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    27a0:	aa0003f6 	mov	x22, x0
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    27a4:	d2800022 	mov	x2, #0x1                   	// #1
    27a8:	d2800081 	mov	x1, #0x4                   	// #4
    27ac:	97fffaf9 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    27b0:	9101e3e4 	add	x4, sp, #0x78
    27b4:	d2c00025 	mov	x5, #0x100000000           	// #4294967296
    27b8:	910263e1 	add	x1, sp, #0x98
    27bc:	52800003 	mov	w3, #0x0                   	// #0
    27c0:	52800002 	mov	w2, #0x0                   	// #0
    27c4:	90000000 	adrp	x0, 2000 <sptAppendValueVector+0x3c>
    27c8:	910f4000 	add	x0, x0, #0x3d0
    27cc:	a909dbe4 	stp	x4, x22, [sp, #152]
    27d0:	f90057e5 	str	x5, [sp, #168]
    27d4:	97fffb13 	bl	1420 <GOMP_parallel@plt>
		free(ubuf);
    27d8:	aa1603e0 	mov	x0, x22
    27dc:	97fffaf1 	bl	13a0 <free@plt>
	sptNewSparseTensor(tsr, nmodes, dims);
    27e0:	b94077e1 	ldr	w1, [sp, #116]
    27e4:	aa1403e2 	mov	x2, x20
    27e8:	aa1303e0 	mov	x0, x19
    27ec:	97fffd51 	bl	1d30 <sptNewSparseTensor>
	for(sptIndex m=0; m < nmodes; ++m) {
    27f0:	b94077e0 	ldr	w0, [sp, #116]
	tsr->nnz = nnz;
    27f4:	f9403fe1 	ldr	x1, [sp, #120]
    27f8:	f9000e61 	str	x1, [x19, #24]
	for(sptIndex m=0; m < nmodes; ++m) {
    27fc:	340011e0 	cbz	w0, 2a38 <sptLoadSparseTensor+0x3d8>
    2800:	52800014 	mov	w20, #0x0                   	// #0
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2804:	52800316 	mov	w22, #0x18                  	// #24
    2808:	14000006 	b	2820 <sptLoadSparseTensor+0x1c0>
	for(sptIndex m=0; m < nmodes; ++m) {
    280c:	b94077e0 	ldr	w0, [sp, #116]
    2810:	11000694 	add	w20, w20, #0x1
	result = sptResizeValueVector(&tsr->values, nnz);
    2814:	f9403fe1 	ldr	x1, [sp, #120]
	for(sptIndex m=0; m < nmodes; ++m) {
    2818:	6b00029f 	cmp	w20, w0
    281c:	540010e2 	b.cs	2a38 <sptLoadSparseTensor+0x3d8>  // b.hs, b.nlast
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    2820:	f9401260 	ldr	x0, [x19, #32]
    2824:	9bb60280 	umaddl	x0, w20, w22, x0
    2828:	97fffeab 	bl	22d4 <sptResizeIndexVector>
		spt_CheckError(result, "SpTns Read", NULL);
    282c:	34ffff00 	cbz	w0, 280c <sptLoadSparseTensor+0x1ac>
    2830:	2a0003e1 	mov	w1, w0
    2834:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2838:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    283c:	91364262 	add	x2, x19, #0xd90
    2840:	913ac000 	add	x0, x0, #0xeb0
    2844:	d2800004 	mov	x4, #0x0                   	// #0
    2848:	528020a3 	mov	w3, #0x105                 	// #261
    284c:	94000739 	bl	4530 <spt_ComplainError>
			iores = p_tt_read_file(tsr, start_index, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
			break;
		case 1:
			iores = p_tt_read_binary_file(tsr, fp);
			spt_CheckOSError(iores != 0, "SpTns Load");
    2850:	97fffaf0 	bl	1410 <__errno_location@plt>
    2854:	b9400016 	ldr	w22, [x0]
    2858:	aa0003f4 	mov	x20, x0
    285c:	314042df 	cmn	w22, #0x10, lsl #12
    2860:	54002321 	b.ne	2cc4 <sptLoadSparseTensor+0x664>  // b.any
    2864:	d503201f 	nop
			break;
	}

	fclose(fp);
    2868:	aa1503e0 	mov	x0, x21
    286c:	97fffa71 	bl	1230 <fclose@plt>

	return 0;
    2870:	52800000 	mov	w0, #0x0                   	// #0
}
    2874:	a94153f3 	ldp	x19, x20, [sp, #16]
    2878:	a9425bf5 	ldp	x21, x22, [sp, #32]
    287c:	a94363f7 	ldp	x23, x24, [sp, #48]
    2880:	a9446bf9 	ldp	x25, x26, [sp, #64]
    2884:	a8cb7bfd 	ldp	x29, x30, [sp], #176
    2888:	d65f03c0 	ret
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    288c:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    2890:	d0000001 	adrp	x1, 4000 <sptNewTimer>
    2894:	aa1703e2 	mov	x2, x23
    2898:	9134e021 	add	x1, x1, #0xd38
    289c:	f947d800 	ldr	x0, [x0, #4016]
    28a0:	f9400000 	ldr	x0, [x0]
    28a4:	97fffae7 	bl	1440 <fprintf@plt>
	iores = fscanf(fp, "%u", &tsr->nmodes);
    28a8:	d0000017 	adrp	x23, 4000 <sptNewTimer>
    28ac:	aa1303e2 	mov	x2, x19
    28b0:	912a42e1 	add	x1, x23, #0xa90
    28b4:	aa1503e0 	mov	x0, x21
    28b8:	97fffa6e 	bl	1270 <__isoc99_fscanf@plt>
	spt_CheckOSError(iores < 0, "SpTns Load");
    28bc:	37f81de0 	tbnz	w0, #31, 2c78 <sptLoadSparseTensor+0x618>
	tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    28c0:	b9400279 	ldr	w25, [x19]
    28c4:	d2800021 	mov	x1, #0x1                   	// #1
    28c8:	d37e7f34 	ubfiz	x20, x25, #2, #32
    28cc:	aa1403e0 	mov	x0, x20
    28d0:	97fffa7c 	bl	12c0 <calloc@plt>
    28d4:	f9000660 	str	x0, [x19, #8]
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    28d8:	b40021a0 	cbz	x0, 2d0c <sptLoadSparseTensor+0x6ac>
	tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    28dc:	aa1403e0 	mov	x0, x20
    28e0:	97fffa60 	bl	1260 <malloc@plt>
    28e4:	f9000a60 	str	x0, [x19, #16]
    28e8:	aa0003e2 	mov	x2, x0
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    28ec:	b4002360 	cbz	x0, 2d58 <sptLoadSparseTensor+0x6f8>
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    28f0:	912a42f8 	add	x24, x23, #0xa90
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    28f4:	52800014 	mov	w20, #0x0                   	// #0
    28f8:	d2800000 	mov	x0, #0x0                   	// #0
    28fc:	350000f9 	cbnz	w25, 2918 <sptLoadSparseTensor+0x2b8>
    2900:	14000020 	b	2980 <sptLoadSparseTensor+0x320>
    2904:	b9400260 	ldr	w0, [x19]
    2908:	11000694 	add	w20, w20, #0x1
    290c:	6b00029f 	cmp	w20, w0
    2910:	54000362 	b.cs	297c <sptLoadSparseTensor+0x31c>  // b.hs, b.nlast
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2914:	f9400a62 	ldr	x2, [x19, #16]
    2918:	8b344842 	add	x2, x2, w20, uxtw #2
    291c:	aa1803e1 	mov	x1, x24
    2920:	aa1503e0 	mov	x0, x21
    2924:	97fffa53 	bl	1270 <__isoc99_fscanf@plt>
		spt_CheckOSError(iores != 1, "SpTns Load");
    2928:	7100041f 	cmp	w0, #0x1
    292c:	54fffec0 	b.eq	2904 <sptLoadSparseTensor+0x2a4>  // b.none
    2930:	97fffab8 	bl	1410 <__errno_location@plt>
    2934:	b940001a 	ldr	w26, [x0]
    2938:	aa0003f9 	mov	x25, x0
    293c:	3140435f 	cmn	w26, #0x10, lsl #12
    2940:	54fffe20 	b.eq	2904 <sptLoadSparseTensor+0x2a4>  // b.none
    2944:	2a1a03e0 	mov	w0, w26
    2948:	97fffa72 	bl	1310 <strerror@plt>
    294c:	11404341 	add	w1, w26, #0x10, lsl #12
    2950:	aa0003e4 	mov	x4, x0
    2954:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2958:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    295c:	91364262 	add	x2, x19, #0xd90
    2960:	9136e280 	add	x0, x20, #0xdb8
    2964:	52800a63 	mov	w3, #0x53                  	// #83
    2968:	940006f2 	bl	4530 <spt_ComplainError>
    296c:	b9400320 	ldr	w0, [x25]
    2970:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2974:	34fff7a0 	cbz	w0, 2868 <sptLoadSparseTensor+0x208>
    2978:	14000020 	b	29f8 <sptLoadSparseTensor+0x398>
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    297c:	2a0003e0 	mov	w0, w0
    2980:	8b000400 	add	x0, x0, x0, lsl #1
	tsr->nnz = 0;
    2984:	f9000e7f 	str	xzr, [x19, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2988:	d37df000 	lsl	x0, x0, #3
    298c:	97fffa35 	bl	1260 <malloc@plt>
    2990:	f9001260 	str	x0, [x19, #32]
    2994:	aa0003f8 	mov	x24, x0
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2998:	b40020a0 	cbz	x0, 2dac <sptLoadSparseTensor+0x74c>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    299c:	b9400260 	ldr	w0, [x19]
    29a0:	52800014 	mov	w20, #0x0                   	// #0
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    29a4:	52800319 	mov	w25, #0x18                  	// #24
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    29a8:	350000e0 	cbnz	w0, 29c4 <sptLoadSparseTensor+0x364>
    29ac:	1400006a 	b	2b54 <sptLoadSparseTensor+0x4f4>
    29b0:	b9400260 	ldr	w0, [x19]
    29b4:	11000694 	add	w20, w20, #0x1
    29b8:	6b00029f 	cmp	w20, w0
    29bc:	54000cc2 	b.cs	2b54 <sptLoadSparseTensor+0x4f4>  // b.hs, b.nlast
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    29c0:	f9401278 	ldr	x24, [x19, #32]
    29c4:	9bb96280 	umaddl	x0, w20, w25, x24
    29c8:	d2800002 	mov	x2, #0x0                   	// #0
    29cc:	d2800001 	mov	x1, #0x0                   	// #0
    29d0:	97fffde9 	bl	2174 <sptNewIndexVector>
		spt_CheckError(retval, "SpTns Load", NULL);
    29d4:	34fffee0 	cbz	w0, 29b0 <sptLoadSparseTensor+0x350>
    29d8:	2a0003e1 	mov	w1, w0
    29dc:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    29e0:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    29e4:	91364262 	add	x2, x19, #0xd90
    29e8:	9136e280 	add	x0, x20, #0xdb8
    29ec:	d2800004 	mov	x4, #0x0                   	// #0
    29f0:	52800b43 	mov	w3, #0x5a                  	// #90
    29f4:	940006cf 	bl	4530 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    29f8:	97fffa86 	bl	1410 <__errno_location@plt>
    29fc:	b9400017 	ldr	w23, [x0]
    2a00:	aa0003f6 	mov	x22, x0
    2a04:	314042ff 	cmn	w23, #0x10, lsl #12
    2a08:	54fff300 	b.eq	2868 <sptLoadSparseTensor+0x208>  // b.none
    2a0c:	2a1703e0 	mov	w0, w23
    2a10:	97fffa40 	bl	1310 <strerror@plt>
    2a14:	91364262 	add	x2, x19, #0xd90
    2a18:	aa0003e4 	mov	x4, x0
    2a1c:	114042e1 	add	w1, w23, #0x10, lsl #12
    2a20:	9136e280 	add	x0, x20, #0xdb8
    2a24:	52802483 	mov	w3, #0x124                 	// #292
    2a28:	940006c2 	bl	4530 <spt_ComplainError>
    2a2c:	b94002c0 	ldr	w0, [x22]
    2a30:	11404000 	add	w0, w0, #0x10, lsl #12
    2a34:	17ffff90 	b	2874 <sptLoadSparseTensor+0x214>
	result = sptResizeValueVector(&tsr->values, nnz);
    2a38:	9100a260 	add	x0, x19, #0x28
    2a3c:	97fffd96 	bl	2094 <sptResizeValueVector>
	spt_CheckError(result, "SpTns Read", NULL);
    2a40:	35001dc0 	cbnz	w0, 2df8 <sptLoadSparseTensor+0x798>
	for(sptIndex m=0; m < nmodes; ++m) {
    2a44:	b94077e0 	ldr	w0, [sp, #116]
    2a48:	52800014 	mov	w20, #0x0                   	// #0
    2a4c:	a90573fb 	stp	x27, x28, [sp, #80]
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2a50:	52800316 	mov	w22, #0x18                  	// #24
	for(sptIndex m=0; m < nmodes; ++m) {
    2a54:	34000180 	cbz	w0, 2a84 <sptLoadSparseTensor+0x424>
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2a58:	f9401260 	ldr	x0, [x19, #32]
    2a5c:	aa1503e3 	mov	x3, x21
    2a60:	b9407be1 	ldr	w1, [sp, #120]
    2a64:	f94047e2 	ldr	x2, [sp, #136]
    2a68:	9bb60280 	umaddl	x0, w20, w22, x0
	for(sptIndex m=0; m < nmodes; ++m) {
    2a6c:	11000694 	add	w20, w20, #0x1
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    2a70:	f9400800 	ldr	x0, [x0, #16]
    2a74:	97fffec4 	bl	2584 <fill_binary_idx.isra.0>
	for(sptIndex m=0; m < nmodes; ++m) {
    2a78:	b94077e0 	ldr	w0, [sp, #116]
    2a7c:	6b00029f 	cmp	w20, w0
    2a80:	54fffec3 	b.cc	2a58 <sptLoadSparseTensor+0x3f8>  // b.lo, b.ul, b.last
	if(header->val_width == sizeof(sptValue)) {
    2a84:	f9404be1 	ldr	x1, [sp, #144]
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2a88:	f9401e7a 	ldr	x26, [x19, #56]
    2a8c:	f9403ffc 	ldr	x28, [sp, #120]
	if(header->val_width == sizeof(sptValue)) {
    2a90:	f100103f 	cmp	x1, #0x4
    2a94:	54001300 	b.eq	2cf4 <sptLoadSparseTensor+0x694>  // b.none
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    2a98:	d2a01000 	mov	x0, #0x800000              	// #8388608
    2a9c:	90000019 	adrp	x25, 2000 <sptAppendValueVector+0x3c>
    2aa0:	97fff9f0 	bl	1260 <malloc@plt>
    2aa4:	9113c339 	add	x25, x25, #0x4f0
    2aa8:	aa0003f6 	mov	x22, x0
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    2aac:	2a1c03f7 	mov	w23, w28
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2ab0:	910263f8 	add	x24, sp, #0x98
    2ab4:	52800014 	mov	w20, #0x0                   	// #0
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2ab8:	52a0021b 	mov	w27, #0x100000              	// #1048576
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2abc:	3400027c 	cbz	w28, 2b08 <sptLoadSparseTensor+0x4a8>
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2ac0:	4b1402f3 	sub	w19, w23, w20
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2ac4:	aa1503e3 	mov	x3, x21
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2ac8:	7144027f 	cmp	w19, #0x100, lsl #12
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2acc:	d2800101 	mov	x1, #0x8                   	// #8
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    2ad0:	1a9b9273 	csel	w19, w19, w27, ls  // ls = plast
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    2ad4:	aa1603e0 	mov	x0, x22
    2ad8:	2a1303e2 	mov	w2, w19
    2adc:	97fffa2d 	bl	1390 <fread@plt>
#pragma omp parallel for schedule(static)
    2ae0:	a909dbfa 	stp	x26, x22, [sp, #152]
    2ae4:	aa1803e1 	mov	x1, x24
    2ae8:	aa1903e0 	mov	x0, x25
    2aec:	52800003 	mov	w3, #0x0                   	// #0
    2af0:	52800002 	mov	w2, #0x0                   	// #0
    2af4:	29154ff4 	stp	w20, w19, [sp, #168]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    2af8:	11440294 	add	w20, w20, #0x100, lsl #12
    2afc:	97fffa49 	bl	1420 <GOMP_parallel@plt>
    2b00:	6b1402ff 	cmp	w23, w20
    2b04:	54fffde8 	b.hi	2ac0 <sptLoadSparseTensor+0x460>  // b.pmore
		free(ubuf);
    2b08:	aa1603e0 	mov	x0, x22
    2b0c:	97fffa25 	bl	13a0 <free@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2b10:	a94573fb 	ldp	x27, x28, [sp, #80]
    2b14:	17ffff55 	b	2868 <sptLoadSparseTensor+0x208>
		fread(buffer, sizeof(sptNnzIndex), count, fin);
    2b18:	aa1503e3 	mov	x3, x21
    2b1c:	9101e3e0 	add	x0, sp, #0x78
    2b20:	d2800022 	mov	x2, #0x1                   	// #1
    2b24:	97fffa1b 	bl	1390 <fread@plt>
    2b28:	17ffff2e 	b	27e0 <sptLoadSparseTensor+0x180>
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    2b2c:	900000a1 	adrp	x1, 16000 <__FRAME_END__+0x10088>
    2b30:	d37df003 	lsl	x3, x0, #3
    2b34:	aa0303e2 	mov	x2, x3
    2b38:	f947d820 	ldr	x0, [x1, #4016]
    2b3c:	d0000001 	adrp	x1, 4000 <sptNewTimer>
    2b40:	9138e021 	add	x1, x1, #0xe38
    2b44:	f9400000 	ldr	x0, [x0]
    2b48:	97fffa3e 	bl	1440 <fprintf@plt>
    2b4c:	f94047e2 	ldr	x2, [sp, #136]
    2b50:	17ffff00 	b	2750 <sptLoadSparseTensor+0xf0>
	retval = sptNewValueVector(&tsr->values, 0, 0);
    2b54:	9100a27a 	add	x26, x19, #0x28
    2b58:	d2800002 	mov	x2, #0x0                   	// #0
    2b5c:	aa1a03e0 	mov	x0, x26
    2b60:	d2800001 	mov	x1, #0x0                   	// #0
    2b64:	910263f8 	add	x24, sp, #0x98
    2b68:	97fffce6 	bl	1f00 <sptNewValueVector>
	spt_CheckError(retval, "SpTns Load", NULL);
    2b6c:	35001580 	cbnz	w0, 2e1c <sptLoadSparseTensor+0x7bc>
			iores = fscanf(fp, "%u", &index);
    2b70:	912a42f7 	add	x23, x23, #0xa90
    2b74:	a90573fb 	stp	x27, x28, [sp, #80]
			iores = fscanf(fp, "%lf", &value);
    2b78:	d000001b 	adrp	x27, 4000 <sptNewTimer>
    2b7c:	9137837b 	add	x27, x27, #0xde0
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2b80:	52800319 	mov	w25, #0x18                  	// #24
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2b84:	b9400260 	ldr	w0, [x19]
    2b88:	52800014 	mov	w20, #0x0                   	// #0
    2b8c:	350001a0 	cbnz	w0, 2bc0 <sptLoadSparseTensor+0x560>
    2b90:	14000021 	b	2c14 <sptLoadSparseTensor+0x5b4>
			if(index < start_index) {
    2b94:	b9409be1 	ldr	w1, [sp, #152]
    2b98:	6b0102df 	cmp	w22, w1
    2b9c:	54000588 	b.hi	2c4c <sptLoadSparseTensor+0x5ec>  // b.pmore
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2ba0:	f9401260 	ldr	x0, [x19, #32]
    2ba4:	4b160021 	sub	w1, w1, w22
    2ba8:	9bb90280 	umaddl	x0, w20, w25, x0
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2bac:	11000694 	add	w20, w20, #0x1
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    2bb0:	97fffd98 	bl	2210 <sptAppendIndexVector>
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    2bb4:	b9400260 	ldr	w0, [x19]
    2bb8:	6b00029f 	cmp	w20, w0
    2bbc:	540002c2 	b.cs	2c14 <sptLoadSparseTensor+0x5b4>  // b.hs, b.nlast
			iores = fscanf(fp, "%u", &index);
    2bc0:	aa1803e2 	mov	x2, x24
    2bc4:	aa1703e1 	mov	x1, x23
    2bc8:	aa1503e0 	mov	x0, x21
    2bcc:	97fff9a9 	bl	1270 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2bd0:	7100041f 	cmp	w0, #0x1
    2bd4:	54fffe00 	b.eq	2b94 <sptLoadSparseTensor+0x534>  // b.none
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2bd8:	b9400260 	ldr	w0, [x19]
    2bdc:	34000180 	cbz	w0, 2c0c <sptLoadSparseTensor+0x5ac>
    2be0:	a9418a63 	ldp	x3, x2, [x19, #24]
    2be4:	51000400 	sub	w0, w0, #0x1
    2be8:	52800304 	mov	w4, #0x18                  	// #24
    2bec:	91006041 	add	x1, x2, #0x18
    2bf0:	9ba40400 	umaddl	x0, w0, w4, x1
    2bf4:	14000002 	b	2bfc <sptLoadSparseTensor+0x59c>
    2bf8:	91006021 	add	x1, x1, #0x18
		tsr->inds[mode].len = tsr->nnz;
    2bfc:	f9000043 	str	x3, [x2]
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2c00:	aa0103e2 	mov	x2, x1
    2c04:	eb01001f 	cmp	x0, x1
    2c08:	54ffff81 	b.ne	2bf8 <sptLoadSparseTensor+0x598>  // b.any
    2c0c:	a94573fb 	ldp	x27, x28, [sp, #80]
    2c10:	17ffff16 	b	2868 <sptLoadSparseTensor+0x208>
			iores = fscanf(fp, "%lf", &value);
    2c14:	aa1803e2 	mov	x2, x24
    2c18:	aa1b03e1 	mov	x1, x27
    2c1c:	aa1503e0 	mov	x0, x21
    2c20:	97fff994 	bl	1270 <__isoc99_fscanf@plt>
			if(iores != 1) {
    2c24:	7100041f 	cmp	w0, #0x1
    2c28:	54fffd81 	b.ne	2bd8 <sptLoadSparseTensor+0x578>  // b.any
			sptAppendValueVector(&tsr->values, value);
    2c2c:	fd404fe0 	ldr	d0, [sp, #152]
    2c30:	aa1a03e0 	mov	x0, x26
    2c34:	1e624000 	fcvt	s0, d0
    2c38:	97fffce3 	bl	1fc4 <sptAppendValueVector>
			++tsr->nnz;
    2c3c:	f9400e60 	ldr	x0, [x19, #24]
    2c40:	91000400 	add	x0, x0, #0x1
    2c44:	f9000e60 	str	x0, [x19, #24]
	while(retval == 0) {
    2c48:	17ffffcf 	b	2b84 <sptLoadSparseTensor+0x524>
				spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
    2c4c:	d0000004 	adrp	x4, 4000 <sptNewTimer>
    2c50:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2c54:	91372084 	add	x4, x4, #0xdc8
    2c58:	91364262 	add	x2, x19, #0xd90
    2c5c:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    2c60:	52800d03 	mov	w3, #0x68                  	// #104
    2c64:	9136e280 	add	x0, x20, #0xdb8
    2c68:	52800061 	mov	w1, #0x3                   	// #3
    2c6c:	94000631 	bl	4530 <spt_ComplainError>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2c70:	a94573fb 	ldp	x27, x28, [sp, #80]
    2c74:	17ffff61 	b	29f8 <sptLoadSparseTensor+0x398>
	spt_CheckOSError(iores < 0, "SpTns Load");
    2c78:	97fff9e6 	bl	1410 <__errno_location@plt>
    2c7c:	b9400014 	ldr	w20, [x0]
    2c80:	aa0003f8 	mov	x24, x0
    2c84:	3140429f 	cmn	w20, #0x10, lsl #12
    2c88:	54ffe1c0 	b.eq	28c0 <sptLoadSparseTensor+0x260>  // b.none
    2c8c:	2a1403e0 	mov	w0, w20
    2c90:	97fff9a0 	bl	1310 <strerror@plt>
    2c94:	11404281 	add	w1, w20, #0x10, lsl #12
    2c98:	aa0003e4 	mov	x4, x0
    2c9c:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2ca0:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    2ca4:	91364262 	add	x2, x19, #0xd90
    2ca8:	9136e280 	add	x0, x20, #0xdb8
    2cac:	52800943 	mov	w3, #0x4a                  	// #74
    2cb0:	94000620 	bl	4530 <spt_ComplainError>
    2cb4:	b9400300 	ldr	w0, [x24]
    2cb8:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2cbc:	34ffdd60 	cbz	w0, 2868 <sptLoadSparseTensor+0x208>
    2cc0:	17ffff4e 	b	29f8 <sptLoadSparseTensor+0x398>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2cc4:	2a1603e0 	mov	w0, w22
    2cc8:	97fff992 	bl	1310 <strerror@plt>
    2ccc:	91364262 	add	x2, x19, #0xd90
    2cd0:	aa0003e4 	mov	x4, x0
    2cd4:	114042c1 	add	w1, w22, #0x10, lsl #12
    2cd8:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    2cdc:	52802503 	mov	w3, #0x128                 	// #296
    2ce0:	9136e000 	add	x0, x0, #0xdb8
    2ce4:	94000613 	bl	4530 <spt_ComplainError>
    2ce8:	b9400280 	ldr	w0, [x20]
    2cec:	11404000 	add	w0, w0, #0x10, lsl #12
    2cf0:	17fffee1 	b	2874 <sptLoadSparseTensor+0x214>
		fread(buffer, sizeof(sptValue), count, fin);
    2cf4:	2a1c03e2 	mov	w2, w28
    2cf8:	aa1a03e0 	mov	x0, x26
    2cfc:	aa1503e3 	mov	x3, x21
    2d00:	97fff9a4 	bl	1390 <fread@plt>
			spt_CheckOSError(iores != 0, "SpTns Load");
    2d04:	a94573fb 	ldp	x27, x28, [sp, #80]
    2d08:	17fffed8 	b	2868 <sptLoadSparseTensor+0x208>
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    2d0c:	97fff9c1 	bl	1410 <__errno_location@plt>
    2d10:	b940001a 	ldr	w26, [x0]
    2d14:	aa0003f8 	mov	x24, x0
    2d18:	3140435f 	cmn	w26, #0x10, lsl #12
    2d1c:	54ffde00 	b.eq	28dc <sptLoadSparseTensor+0x27c>  // b.none
    2d20:	2a1a03e0 	mov	w0, w26
    2d24:	97fff97b 	bl	1310 <strerror@plt>
    2d28:	11404341 	add	w1, w26, #0x10, lsl #12
    2d2c:	aa0003e4 	mov	x4, x0
    2d30:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2d34:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    2d38:	91364262 	add	x2, x19, #0xd90
    2d3c:	9136e280 	add	x0, x20, #0xdb8
    2d40:	528009a3 	mov	w3, #0x4d                  	// #77
    2d44:	940005fb 	bl	4530 <spt_ComplainError>
    2d48:	b9400300 	ldr	w0, [x24]
    2d4c:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2d50:	34ffd8c0 	cbz	w0, 2868 <sptLoadSparseTensor+0x208>
    2d54:	17ffff29 	b	29f8 <sptLoadSparseTensor+0x398>
    2d58:	f90037e0 	str	x0, [sp, #104]
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    2d5c:	97fff9ad 	bl	1410 <__errno_location@plt>
    2d60:	b9400014 	ldr	w20, [x0]
    2d64:	aa0003f8 	mov	x24, x0
    2d68:	f94037e2 	ldr	x2, [sp, #104]
    2d6c:	3140429f 	cmn	w20, #0x10, lsl #12
    2d70:	54ffdc00 	b.eq	28f0 <sptLoadSparseTensor+0x290>  // b.none
    2d74:	2a1403e0 	mov	w0, w20
    2d78:	97fff966 	bl	1310 <strerror@plt>
    2d7c:	11404281 	add	w1, w20, #0x10, lsl #12
    2d80:	aa0003e4 	mov	x4, x0
    2d84:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2d88:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    2d8c:	91364262 	add	x2, x19, #0xd90
    2d90:	9136e280 	add	x0, x20, #0xdb8
    2d94:	52800a03 	mov	w3, #0x50                  	// #80
    2d98:	940005e6 	bl	4530 <spt_ComplainError>
    2d9c:	b9400300 	ldr	w0, [x24]
    2da0:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2da4:	34ffd620 	cbz	w0, 2868 <sptLoadSparseTensor+0x208>
    2da8:	17ffff14 	b	29f8 <sptLoadSparseTensor+0x398>
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2dac:	97fff999 	bl	1410 <__errno_location@plt>
    2db0:	b9400014 	ldr	w20, [x0]
    2db4:	aa0003f9 	mov	x25, x0
    2db8:	3140429f 	cmn	w20, #0x10, lsl #12
    2dbc:	54ffdf00 	b.eq	299c <sptLoadSparseTensor+0x33c>  // b.none
    2dc0:	2a1403e0 	mov	w0, w20
    2dc4:	97fff953 	bl	1310 <strerror@plt>
    2dc8:	11404281 	add	w1, w20, #0x10, lsl #12
    2dcc:	aa0003e4 	mov	x4, x0
    2dd0:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2dd4:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    2dd8:	91364262 	add	x2, x19, #0xd90
    2ddc:	9136e280 	add	x0, x20, #0xdb8
    2de0:	52800ae3 	mov	w3, #0x57                  	// #87
    2de4:	940005d3 	bl	4530 <spt_ComplainError>
    2de8:	b9400320 	ldr	w0, [x25]
    2dec:	11404000 	add	w0, w0, #0x10, lsl #12
			spt_CheckOSError(iores != 0, "SpTns Load");
    2df0:	34ffd3c0 	cbz	w0, 2868 <sptLoadSparseTensor+0x208>
    2df4:	17ffff01 	b	29f8 <sptLoadSparseTensor+0x398>
	spt_CheckError(result, "SpTns Read", NULL);
    2df8:	2a0003e1 	mov	w1, w0
    2dfc:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2e00:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    2e04:	91364262 	add	x2, x19, #0xd90
    2e08:	913ac000 	add	x0, x0, #0xeb0
    2e0c:	d2800004 	mov	x4, #0x0                   	// #0
    2e10:	52802103 	mov	w3, #0x108                 	// #264
    2e14:	940005c7 	bl	4530 <spt_ComplainError>
    2e18:	17fffe8e 	b	2850 <sptLoadSparseTensor+0x1f0>
	spt_CheckError(retval, "SpTns Load", NULL);
    2e1c:	2a0003e1 	mov	w1, w0
    2e20:	d0000013 	adrp	x19, 4000 <sptNewTimer>
    2e24:	d0000014 	adrp	x20, 4000 <sptNewTimer>
    2e28:	91364262 	add	x2, x19, #0xd90
    2e2c:	9136e280 	add	x0, x20, #0xdb8
    2e30:	d2800004 	mov	x4, #0x0                   	// #0
    2e34:	52800ba3 	mov	w3, #0x5d                  	// #93
    2e38:	940005be 	bl	4530 <spt_ComplainError>
    2e3c:	17fffeef 	b	29f8 <sptLoadSparseTensor+0x398>
	sptAssert(fp != NULL);
    2e40:	12800000 	mov	w0, #0xffffffff            	// #-1
    2e44:	a90573fb 	stp	x27, x28, [sp, #80]
    2e48:	97fff8de 	bl	11c0 <exit@plt>
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    2e4c:	900000a0 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    2e50:	d37df043 	lsl	x3, x2, #3
    2e54:	aa0303e2 	mov	x2, x3
    2e58:	d0000001 	adrp	x1, 4000 <sptNewTimer>
    2e5c:	f947d800 	ldr	x0, [x0, #4016]
    2e60:	9137a021 	add	x1, x1, #0xde8
    2e64:	a90573fb 	stp	x27, x28, [sp, #80]
    2e68:	f9400000 	ldr	x0, [x0]
    2e6c:	97fff975 	bl	1440 <fprintf@plt>
		exit(-1);
    2e70:	12800000 	mov	w0, #0xffffffff            	// #-1
    2e74:	97fff8d3 	bl	11c0 <exit@plt>
    2e78:	d503201f 	nop
    2e7c:	d503201f 	nop

0000000000002e80 <sptNewMatrix>:
 */
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
	mtx->nrows = nrows;
	mtx->ncols = ncols;
	mtx->cap = nrows != 0 ? nrows : 1;
	mtx->stride = ((ncols-1)/8+1)*8;
    2e80:	51000443 	sub	w3, w2, #0x1
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    2e84:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2e88:	7100003f 	cmp	w1, #0x0
	mtx->stride = ((ncols-1)/8+1)*8;
    2e8c:	121d7063 	and	w3, w3, #0xfffffff8
    2e90:	1a9f1424 	csinc	w4, w1, wzr, ne  // ne = any
    2e94:	11002063 	add	w3, w3, #0x8
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    2e98:	910003fd 	mov	x29, sp
    2e9c:	a90153f3 	stp	x19, x20, [sp, #16]
    2ea0:	aa0003f3 	mov	x19, x0
#ifdef _ISOC11_SOURCE
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    2ea4:	1b047c74 	mul	w20, w3, w4
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    2ea8:	a9025bf5 	stp	x21, x22, [sp, #32]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    2eac:	d2800400 	mov	x0, #0x20                  	// #32
	mtx->ncols = ncols;
    2eb0:	29000a61 	stp	w1, w2, [x19]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    2eb4:	d37ef694 	lsl	x20, x20, #2
	mtx->stride = ((ncols-1)/8+1)*8;
    2eb8:	29010e64 	stp	w4, w3, [x19, #8]
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    2ebc:	aa1403e1 	mov	x1, x20
    2ec0:	97fff8f8 	bl	12a0 <aligned_alloc@plt>
    2ec4:	f9000a60 	str	x0, [x19, #16]
    2ec8:	aa0003f5 	mov	x21, x0
		}
	}
#else
	mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
	spt_CheckOSError(!mtx->values, "Mtx New");
    2ecc:	b4000140 	cbz	x0, 2ef4 <sptNewMatrix+0x74>
	memset(mtx->values, 0, mtx->cap * mtx->stride * sizeof (sptValue));
    2ed0:	aa1403e2 	mov	x2, x20
    2ed4:	aa1503e0 	mov	x0, x21
    2ed8:	52800001 	mov	w1, #0x0                   	// #0
    2edc:	97fff8f5 	bl	12b0 <memset@plt>
	return 0;
    2ee0:	52800000 	mov	w0, #0x0                   	// #0
}
    2ee4:	a94153f3 	ldp	x19, x20, [sp, #16]
    2ee8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    2eec:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2ef0:	d65f03c0 	ret
	spt_CheckOSError(!mtx->values, "Mtx New");
    2ef4:	97fff947 	bl	1410 <__errno_location@plt>
    2ef8:	b9400016 	ldr	w22, [x0]
    2efc:	aa0003f3 	mov	x19, x0
    2f00:	314042df 	cmn	w22, #0x10, lsl #12
    2f04:	54fffe60 	b.eq	2ed0 <sptNewMatrix+0x50>  // b.none
    2f08:	2a1603e0 	mov	w0, w22
    2f0c:	97fff901 	bl	1310 <strerror@plt>
    2f10:	114042c1 	add	w1, w22, #0x10, lsl #12
    2f14:	aa0003e4 	mov	x4, x0
    2f18:	d0000002 	adrp	x2, 4000 <sptNewTimer>
    2f1c:	d0000000 	adrp	x0, 4000 <sptNewTimer>
    2f20:	913b4042 	add	x2, x2, #0xed0
    2f24:	913be000 	add	x0, x0, #0xef8
    2f28:	52800703 	mov	w3, #0x38                  	// #56
    2f2c:	94000581 	bl	4530 <spt_ComplainError>
    2f30:	b9400260 	ldr	w0, [x19]
    2f34:	11404000 	add	w0, w0, #0x10, lsl #12
    2f38:	17ffffeb 	b	2ee4 <sptNewMatrix+0x64>
    2f3c:	d503201f 	nop

0000000000002f40 <sptRandomizeMatrix>:
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, bool random) {
    2f40:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    2f44:	910003fd 	mov	x29, sp
    2f48:	a9025bf5 	stp	x21, x22, [sp, #32]
    2f4c:	12001c35 	and	w21, w1, #0xff
	for(sptIndex i=0; i<mtx->nrows; ++i)
    2f50:	b9400001 	ldr	w1, [x0]
    2f54:	34000601 	cbz	w1, 3014 <sptRandomizeMatrix+0xd4>
    2f58:	a90363f7 	stp	x23, x24, [sp, #48]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
			if (random) {
				srand(time(NULL) + (rand()%100) + i + j);
    2f5c:	5290a3f8 	mov	w24, #0x851f                	// #34079
    2f60:	72aa3d78 	movk	w24, #0x51eb, lsl #16
    2f64:	52800c97 	mov	w23, #0x64                  	// #100
    2f68:	a90153f3 	stp	x19, x20, [sp, #16]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    2f6c:	52800014 	mov	w20, #0x0                   	// #0
    2f70:	a9046bf9 	stp	x25, x26, [sp, #64]
    2f74:	aa0003fa 	mov	x26, x0
    2f78:	b9400400 	ldr	w0, [x0, #4]
    2f7c:	f9002bfb 	str	x27, [sp, #80]
			} else {
				srand(1234 + i + j);
    2f80:	11134a96 	add	w22, w20, #0x4d2
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    2f84:	52800013 	mov	w19, #0x0                   	// #0
    2f88:	350002a0 	cbnz	w0, 2fdc <sptRandomizeMatrix+0x9c>
    2f8c:	1400001b 	b	2ff8 <sptRandomizeMatrix+0xb8>
				srand(time(NULL) + (rand()%100) + i + j);
    2f90:	97fff8b0 	bl	1250 <time@plt>
    2f94:	aa0003f9 	mov	x25, x0
    2f98:	97fff8d2 	bl	12e0 <rand@plt>
    2f9c:	9b387c01 	smull	x1, w0, w24
    2fa0:	9365fc21 	asr	x1, x1, #37
    2fa4:	4b807c21 	sub	w1, w1, w0, asr #31
    2fa8:	1b178021 	msub	w1, w1, w23, w0
    2fac:	0b190020 	add	w0, w1, w25
    2fb0:	0b1b0000 	add	w0, w0, w27
    2fb4:	97fff90b 	bl	13e0 <srand@plt>
			}
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    2fb8:	b9400f59 	ldr	w25, [x26, #12]
    2fbc:	f9400b5b 	ldr	x27, [x26, #16]
    2fc0:	940005d4 	bl	4710 <sptRandomValue>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    2fc4:	b9400740 	ldr	w0, [x26, #4]
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    2fc8:	1b194e99 	madd	w25, w20, w25, w19
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    2fcc:	11000673 	add	w19, w19, #0x1
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    2fd0:	bc397b60 	str	s0, [x27, x25, lsl #2]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    2fd4:	6b13001f 	cmp	w0, w19
    2fd8:	540000e9 	b.ls	2ff4 <sptRandomizeMatrix+0xb4>  // b.plast
				srand(1234 + i + j);
    2fdc:	0b13029b 	add	w27, w20, w19
				srand(time(NULL) + (rand()%100) + i + j);
    2fe0:	d2800000 	mov	x0, #0x0                   	// #0
			if (random) {
    2fe4:	35fffd75 	cbnz	w21, 2f90 <sptRandomizeMatrix+0x50>
				srand(1234 + i + j);
    2fe8:	0b1302c0 	add	w0, w22, w19
    2fec:	97fff8fd 	bl	13e0 <srand@plt>
    2ff0:	17fffff2 	b	2fb8 <sptRandomizeMatrix+0x78>
    2ff4:	b9400341 	ldr	w1, [x26]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    2ff8:	11000694 	add	w20, w20, #0x1
    2ffc:	6b14003f 	cmp	w1, w20
    3000:	54fffc08 	b.hi	2f80 <sptRandomizeMatrix+0x40>  // b.pmore
    3004:	a94153f3 	ldp	x19, x20, [sp, #16]
    3008:	a94363f7 	ldp	x23, x24, [sp, #48]
    300c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3010:	f9402bfb 	ldr	x27, [sp, #80]
		}
	return 0;
}
    3014:	52800000 	mov	w0, #0x0                   	// #0
    3018:	a9425bf5 	ldp	x21, x22, [sp, #32]
    301c:	a8c67bfd 	ldp	x29, x30, [sp], #96
    3020:	d65f03c0 	ret

0000000000003024 <sptConstantMatrix>:
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3024:	b9400006 	ldr	w6, [x0]
    3028:	34000226 	cbz	w6, 306c <sptConstantMatrix+0x48>
		for(sptIndex j=0; j<mtx->ncols; ++j)
    302c:	b9400405 	ldr	w5, [x0, #4]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3030:	52800004 	mov	w4, #0x0                   	// #0
    3034:	d503201f 	nop
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3038:	34000145 	cbz	w5, 3060 <sptConstantMatrix+0x3c>
			mtx->values[i * mtx->stride + j] = val;
    303c:	b9400c01 	ldr	w1, [x0, #12]
    3040:	f9400803 	ldr	x3, [x0, #16]
    3044:	1b017c81 	mul	w1, w4, w1
    3048:	0b050022 	add	w2, w1, w5
    304c:	d503201f 	nop
    3050:	bc215860 	str	s0, [x3, w1, uxtw #2]
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3054:	11000421 	add	w1, w1, #0x1
    3058:	6b01005f 	cmp	w2, w1
    305c:	54ffffa1 	b.ne	3050 <sptConstantMatrix+0x2c>  // b.any
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3060:	11000484 	add	w4, w4, #0x1
    3064:	6b06009f 	cmp	w4, w6
    3068:	54fffe81 	b.ne	3038 <sptConstantMatrix+0x14>  // b.any
	return 0;
}
    306c:	52800000 	mov	w0, #0x0                   	// #0
    3070:	d65f03c0 	ret

0000000000003074 <sptFreeMatrix>:
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    3074:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    3078:	910003fd 	mov	x29, sp
    307c:	f9000bf3 	str	x19, [sp, #16]
    3080:	aa0003f3 	mov	x19, x0
	free(mtx->values);
    3084:	f9400800 	ldr	x0, [x0, #16]
    3088:	97fff8c6 	bl	13a0 <free@plt>
	mtx->nrows = 0;
	mtx->ncols = 0;
	mtx->cap = 0;
    308c:	a9007e7f 	stp	xzr, xzr, [x19]
	mtx->stride = 0;
}
    3090:	f9400bf3 	ldr	x19, [sp, #16]
    3094:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3098:	d65f03c0 	ret
    309c:	d503201f 	nop

00000000000030a0 <sptSparseTensorDensity>:
#include <bits/types/FILE.h>
#include <stdio.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
    30a0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    30a4:	910003fd 	mov	x29, sp
    30a8:	f9000bf3 	str	x19, [sp, #16]
    30ac:	aa0003f3 	mov	x19, x0
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    30b0:	b9400000 	ldr	w0, [x0]
{
    30b4:	fd000fe8 	str	d8, [sp, #24]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    30b8:	1e6e1008 	fmov	d8, #1.000000000000000000e+00
    30bc:	fd400e60 	ldr	d0, [x19, #24]
    30c0:	1e630001 	ucvtf	d1, w0
    30c4:	7e61d800 	ucvtf	d0, d0
    30c8:	1e611901 	fdiv	d1, d8, d1
    30cc:	97fff851 	bl	1210 <pow@plt>
	double density = 1.0;
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    30d0:	b9400262 	ldr	w2, [x19]
    30d4:	34000202 	cbz	w2, 3114 <sptSparseTensorDensity+0x74>
    30d8:	1e604002 	fmov	d2, d0
	double density = 1.0;
    30dc:	1e604100 	fmov	d0, d8
		density *= root / (double)tsr->ndims[m];
    30e0:	f9400a63 	ldr	x3, [x19, #16]
    30e4:	d2800000 	mov	x0, #0x0                   	// #0
    30e8:	b8607861 	ldr	w1, [x3, x0, lsl #2]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    30ec:	91000400 	add	x0, x0, #0x1
		density *= root / (double)tsr->ndims[m];
    30f0:	1e630021 	ucvtf	d1, w1
    30f4:	1e611841 	fdiv	d1, d2, d1
    30f8:	1e610800 	fmul	d0, d0, d1
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    30fc:	6b00005f 	cmp	w2, w0
    3100:	54ffff48 	b.hi	30e8 <sptSparseTensorDensity+0x48>  // b.pmore
	}

	return density;
}
    3104:	f9400bf3 	ldr	x19, [sp, #16]
    3108:	fd400fe8 	ldr	d8, [sp, #24]
    310c:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3110:	d65f03c0 	ret
	double density = 1.0;
    3114:	1e604100 	fmov	d0, d8
}
    3118:	fd400fe8 	ldr	d8, [sp, #24]
    311c:	f9400bf3 	ldr	x19, [sp, #16]
    3120:	a8c27bfd 	ldp	x29, x30, [sp], #32
    3124:	d65f03c0 	ret
    3128:	d503201f 	nop
    312c:	d503201f 	nop

0000000000003130 <sptSparseTensorStatus>:



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
    3130:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3134:	aa0103e3 	mov	x3, x1
    3138:	d2800822 	mov	x2, #0x41                  	// #65
{
    313c:	910003fd 	mov	x29, sp
    3140:	a90153f3 	stp	x19, x20, [sp, #16]
    3144:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3148:	b0000000 	adrp	x0, 4000 <sptNewTimer>
    314c:	913c0000 	add	x0, x0, #0xf00
{
    3150:	a9025bf5 	stp	x21, x22, [sp, #32]
    3154:	aa0103f5 	mov	x21, x1
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    3158:	d2800021 	mov	x1, #0x1                   	// #1
    315c:	97fff895 	bl	13b0 <fwrite@plt>
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3160:	f9400a62 	ldr	x2, [x19, #16]
    3164:	aa1503e0 	mov	x0, x21
    3168:	b0000001 	adrp	x1, 4000 <sptNewTimer>
    316c:	913d2021 	add	x1, x1, #0xf48
    3170:	b0000016 	adrp	x22, 4000 <sptNewTimer>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3174:	52800034 	mov	w20, #0x1                   	// #1
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    3178:	b9400042 	ldr	w2, [x2]
    317c:	913e62d6 	add	x22, x22, #0xf98
    3180:	97fff8b0 	bl	1440 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    3184:	b9400260 	ldr	w0, [x19]
    3188:	6b14001f 	cmp	w0, w20
    318c:	54000149 	b.ls	31b4 <sptSparseTensorStatus+0x84>  // b.plast
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    3190:	f9400a62 	ldr	x2, [x19, #16]
    3194:	aa1603e1 	mov	x1, x22
    3198:	aa1503e0 	mov	x0, x21
    319c:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    31a0:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    31a4:	97fff8a7 	bl	1440 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    31a8:	b9400260 	ldr	w0, [x19]
    31ac:	6b14001f 	cmp	w0, w20
    31b0:	54ffff08 	b.hi	3190 <sptSparseTensorStatus+0x60>  // b.pmore
	}
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    31b4:	f9400e62 	ldr	x2, [x19, #24]
    31b8:	b0000001 	adrp	x1, 4000 <sptNewTimer>
    31bc:	913d6021 	add	x1, x1, #0xf58
    31c0:	aa1503e0 	mov	x0, x21
    31c4:	b0000016 	adrp	x22, 4000 <sptNewTimer>
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));

	fprintf(fp, "Average slice length (c): ");
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    31c8:	52800014 	mov	w20, #0x0                   	// #0
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    31cc:	97fff89d 	bl	1440 <fprintf@plt>
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    31d0:	913e82d6 	add	x22, x22, #0xfa0
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));
    31d4:	aa1303e0 	mov	x0, x19
    31d8:	97ffffb2 	bl	30a0 <sptSparseTensorDensity>
    31dc:	aa1503e0 	mov	x0, x21
    31e0:	b0000001 	adrp	x1, 4000 <sptNewTimer>
    31e4:	913da021 	add	x1, x1, #0xf68
    31e8:	97fff896 	bl	1440 <fprintf@plt>
	fprintf(fp, "Average slice length (c): ");
    31ec:	d2800342 	mov	x2, #0x1a                  	// #26
    31f0:	aa1503e3 	mov	x3, x21
    31f4:	b0000000 	adrp	x0, 4000 <sptNewTimer>
    31f8:	d2800021 	mov	x1, #0x1                   	// #1
    31fc:	913de000 	add	x0, x0, #0xf78
    3200:	97fff86c 	bl	13b0 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3204:	b9400260 	ldr	w0, [x19]
    3208:	d2800002 	mov	x2, #0x0                   	// #0
    320c:	7100041f 	cmp	w0, #0x1
    3210:	54000220 	b.eq	3254 <sptSparseTensorStatus+0x124>  // b.none
    3214:	d503201f 	nop
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3218:	f9400a62 	ldr	x2, [x19, #16]
    321c:	aa1603e1 	mov	x1, x22
    3220:	fd400e60 	ldr	d0, [x19, #24]
    3224:	aa1503e0 	mov	x0, x21
    3228:	b8745842 	ldr	w2, [x2, w20, uxtw #2]
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    322c:	11000694 	add	w20, w20, #0x1
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    3230:	7e61d800 	ucvtf	d0, d0
    3234:	1e630041 	ucvtf	d1, w2
    3238:	1e611800 	fdiv	d0, d0, d1
    323c:	97fff881 	bl	1440 <fprintf@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    3240:	b9400262 	ldr	w2, [x19]
    3244:	51000442 	sub	w2, w2, #0x1
    3248:	6b14005f 	cmp	w2, w20
    324c:	54fffe68 	b.hi	3218 <sptSparseTensorStatus+0xe8>  // b.pmore
	}
	fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);
    3250:	d37e7c42 	ubfiz	x2, x2, #2, #32
    3254:	f9400a60 	ldr	x0, [x19, #16]
    3258:	b0000001 	adrp	x1, 4000 <sptNewTimer>
    325c:	fd400e60 	ldr	d0, [x19, #24]
    3260:	913ec021 	add	x1, x1, #0xfb0
    3264:	b8626802 	ldr	w2, [x0, x2]
    3268:	aa1503e0 	mov	x0, x21
    326c:	7e61d800 	ucvtf	d0, d0
    3270:	1e630041 	ucvtf	d1, w2
    3274:	1e611800 	fdiv	d0, d0, d1
    3278:	97fff872 	bl	1440 <fprintf@plt>

	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    327c:	b9400261 	ldr	w1, [x19]
    3280:	f9400e60 	ldr	x0, [x19, #24]
    3284:	9b000020 	madd	x0, x1, x0, x0
    3288:	d37ef400 	lsl	x0, x0, #2
    328c:	940004e5 	bl	4620 <sptBytesString>
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3290:	aa0003e2 	mov	x2, x0
	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    3294:	aa0003f3 	mov	x19, x0
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    3298:	b0000001 	adrp	x1, 4000 <sptNewTimer>
    329c:	aa1503e0 	mov	x0, x21
    32a0:	913ee021 	add	x1, x1, #0xfb8
    32a4:	97fff867 	bl	1440 <fprintf@plt>
	fprintf(fp, "\n");
    32a8:	aa1503e1 	mov	x1, x21
    32ac:	52800140 	mov	w0, #0xa                   	// #10
    32b0:	97fff7cc 	bl	11e0 <fputc@plt>
	free(bytestr);
    32b4:	aa1303e0 	mov	x0, x19
}
    32b8:	a94153f3 	ldp	x19, x20, [sp, #16]
    32bc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    32c0:	a8c37bfd 	ldp	x29, x30, [sp], #48
	free(bytestr);
    32c4:	17fff837 	b	13a0 <free@plt>
    32c8:	d503201f 	nop
    32cc:	d503201f 	nop

00000000000032d0 <sptMTTKRP_3D>:

int sptMTTKRP_3D(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode)
{
    32d0:	a9b87bfd 	stp	x29, x30, [sp, #-128]!
    32d4:	910003fd 	mov	x29, sp
    32d8:	a9046bf9 	stp	x25, x26, [sp, #64]
    32dc:	aa0003f9 	mov	x25, x0
	sptIndex const nmodes = X->nmodes;
	sptNnzIndex const nnz = X->nnz;
	sptIndex const * const ndims = X->ndims;
	sptValue const * const restrict vals = X->values.data;
	sptIndex const stride = mats[0]->stride;
    32e0:	f9400020 	ldr	x0, [x1]
{
    32e4:	a90363f7 	stp	x23, x24, [sp, #48]
    32e8:	aa0103f8 	mov	x24, x1

	/* Check the mats. */
	sptAssert(nmodes ==3);
    32ec:	b9400321 	ldr	w1, [x25]
{
    32f0:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const stride = mats[0]->stride;
    32f4:	b9400c14 	ldr	w20, [x0, #12]
{
    32f8:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptNnzIndex const nnz = X->nnz;
    32fc:	a9415b25 	ldp	x5, x22, [x25, #16]
	sptValue const * const restrict vals = X->values.data;
    3300:	f9401f37 	ldr	x23, [x25, #56]
	sptAssert(nmodes ==3);
    3304:	a90573fb 	stp	x27, x28, [sp, #80]
    3308:	71000c3f 	cmp	w1, #0x3
    330c:	54000f61 	b.ne	34f8 <sptMTTKRP_3D+0x228>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3310:	f9400f06 	ldr	x6, [x24, #24]
    3314:	aa0203fc 	mov	x28, x2
    3318:	b9400401 	ldr	w1, [x0, #4]
    331c:	d2800004 	mov	x4, #0x0                   	// #0
    3320:	b94004c2 	ldr	w2, [x6, #4]
    3324:	6b02003f 	cmp	w1, w2
    3328:	54000181 	b.ne	3358 <sptMTTKRP_3D+0x88>  // b.any
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
		}
		if(mats[i]->nrows != ndims[i]) {
    332c:	b9400001 	ldr	w1, [x0]
    3330:	b86478a0 	ldr	w0, [x5, x4, lsl #2]
    3334:	6b00003f 	cmp	w1, w0
    3338:	54000ca1 	b.ne	34cc <sptMTTKRP_3D+0x1fc>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    333c:	91000484 	add	x4, x4, #0x1
    3340:	f1000c9f 	cmp	x4, #0x3
    3344:	54000200 	b.eq	3384 <sptMTTKRP_3D+0xb4>  // b.none
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3348:	f8647b00 	ldr	x0, [x24, x4, lsl #3]
    334c:	b9400401 	ldr	w1, [x0, #4]
    3350:	6b02003f 	cmp	w1, w2
    3354:	54fffec0 	b.eq	332c <sptMTTKRP_3D+0x5c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3358:	b0000004 	adrp	x4, 4000 <sptNewTimer>
    335c:	b0000002 	adrp	x2, 4000 <sptNewTimer>
    3360:	913f4084 	add	x4, x4, #0xfd0
    3364:	913fe042 	add	x2, x2, #0xff8
    3368:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    336c:	52801263 	mov	w3, #0x93                  	// #147
    3370:	91008000 	add	x0, x0, #0x20
    3374:	52800041 	mov	w1, #0x2                   	// #2
    3378:	9400046e 	bl	4530 <spt_ComplainError>
    337c:	52800040 	mov	w0, #0x2                   	// #2
    3380:	1400004c 	b	34b0 <sptMTTKRP_3D+0x1e0>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
		}
	}


	sptIndex const tmpI = mats[mode]->nrows;
    3384:	f8635b00 	ldr	x0, [x24, w3, uxtw #3]
	sptIndex const R = mats[mode]->ncols;
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3388:	5280031b 	mov	w27, #0x18                  	// #24
    338c:	f9401324 	ldr	x4, [x25, #32]
	sptMatrix * const restrict M = mats[nmodes];
	sptValue * const restrict mvals = M->values;
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3390:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    3394:	29405402 	ldp	w2, w21, [x0]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3398:	9bbb1063 	umaddl	x3, w3, w27, x4
	sptValue * const restrict mvals = M->values;
    339c:	f94008d3 	ldr	x19, [x6, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    33a0:	1b027e82 	mul	w2, w20, w2
    33a4:	aa1303e0 	mov	x0, x19
    33a8:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    33ac:	f940087a 	ldr	x26, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    33b0:	fd0033e8 	str	d8, [sp, #96]
    33b4:	97fff7bf 	bl	12b0 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
	sptIndex times_mat_index_2 = mats_order[2];
    33b8:	29409382 	ldp	w2, w4, [x28, #4]
	sptIndex mode_i;
	sptIndex tmp_i_1, tmp_i_2;
	sptValue entry;

	sptTimer timer;
	sptNewTimer(&timer, 0);
    33bc:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    33c0:	f9401325 	ldr	x5, [x25, #32]
	sptNewTimer(&timer, 0);
    33c4:	9101e3e0 	add	x0, sp, #0x78
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    33c8:	f8625b1c 	ldr	x28, [x24, w2, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    33cc:	9bbb1443 	umaddl	x3, w2, w27, x5
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    33d0:	9bbb1482 	umaddl	x2, w4, w27, x5
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    33d4:	f8645b1b 	ldr	x27, [x24, w4, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    33d8:	f9400879 	ldr	x25, [x3, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    33dc:	f9400858 	ldr	x24, [x2, #16]
	sptNewTimer(&timer, 0);
    33e0:	94000308 	bl	4000 <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    33e4:	f9403fe0 	ldr	x0, [sp, #120]
    33e8:	94000326 	bl	4080 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    33ec:	d2800009 	mov	x9, #0x0                   	// #0
    33f0:	b40003f6 	cbz	x22, 346c <sptMTTKRP_3D+0x19c>
    33f4:	d503201f 	nop
		mode_i = mode_ind[x];
    33f8:	b8697b44 	ldr	w4, [x26, x9, lsl #2]
		tmp_i_1 = times_inds_1[x];
    33fc:	b8697b26 	ldr	w6, [x25, x9, lsl #2]
		tmp_i_2 = times_inds_2[x];
    3400:	b8697b05 	ldr	w5, [x24, x9, lsl #2]
		entry = vals[x];
    3404:	bc697ae3 	ldr	s3, [x23, x9, lsl #2]

		for(sptIndex r=0; r<R; ++r) {
    3408:	340002d5 	cbz	w21, 3460 <sptMTTKRP_3D+0x190>
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    340c:	1b047e84 	mul	w4, w20, w4
    3410:	1b067e86 	mul	w6, w20, w6
    3414:	1b057e85 	mul	w5, w20, w5
    3418:	2a0403e0 	mov	w0, w4
    341c:	f9400b67 	ldr	x7, [x27, #16]
    3420:	4b0400c6 	sub	w6, w6, w4
    3424:	f9400b88 	ldr	x8, [x28, #16]
    3428:	4b0400a5 	sub	w5, w5, w4
    342c:	0b150084 	add	w4, w4, w21
    3430:	0b0000c3 	add	w3, w6, w0
    3434:	0b0000a2 	add	w2, w5, w0
    3438:	2a0003e1 	mov	w1, w0
		for(sptIndex r=0; r<R; ++r) {
    343c:	11000400 	add	w0, w0, #0x1
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3440:	bc637901 	ldr	s1, [x8, x3, lsl #2]
    3444:	bc6278e2 	ldr	s2, [x7, x2, lsl #2]
    3448:	bc617a60 	ldr	s0, [x19, x1, lsl #2]
    344c:	1e210861 	fmul	s1, s3, s1
    3450:	1f020020 	fmadd	s0, s1, s2, s0
    3454:	bc217a60 	str	s0, [x19, x1, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3458:	6b04001f 	cmp	w0, w4
    345c:	54fffea1 	b.ne	3430 <sptMTTKRP_3D+0x160>  // b.any
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3460:	91000529 	add	x9, x9, #0x1
    3464:	eb0902df 	cmp	x22, x9
    3468:	54fffc81 	b.ne	33f8 <sptMTTKRP_3D+0x128>  // b.any
		}
	}
	sptStopTimer(timer);
    346c:	f9403fe0 	ldr	x0, [sp, #120]
    3470:	94000334 	bl	4140 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    3474:	f9403fe0 	ldr	x0, [sp, #120]
    3478:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    347c:	91008021 	add	x1, x1, #0x20
    3480:	94000370 	bl	4240 <sptPrintElapsedTime>
    3484:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3488:	f9403fe0 	ldr	x0, [sp, #120]
    348c:	940003a9 	bl	4330 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3490:	1e604100 	fmov	d0, d8
    3494:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3498:	91016000 	add	x0, x0, #0x58
    349c:	97fff7d9 	bl	1400 <printf@plt>
	printf("\n");
    34a0:	52800140 	mov	w0, #0xa                   	// #10
    34a4:	97fff7e3 	bl	1430 <putchar@plt>

	return 0;
    34a8:	fd4033e8 	ldr	d8, [sp, #96]
    34ac:	52800000 	mov	w0, #0x0                   	// #0
}
    34b0:	a94153f3 	ldp	x19, x20, [sp, #16]
    34b4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    34b8:	a94363f7 	ldp	x23, x24, [sp, #48]
    34bc:	a9446bf9 	ldp	x25, x26, [sp, #64]
    34c0:	a94573fb 	ldp	x27, x28, [sp, #80]
    34c4:	a8c87bfd 	ldp	x29, x30, [sp], #128
    34c8:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    34cc:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    34d0:	b0000002 	adrp	x2, 4000 <sptNewTimer>
    34d4:	9100e084 	add	x4, x4, #0x38
    34d8:	913fe042 	add	x2, x2, #0xff8
    34dc:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    34e0:	528012c3 	mov	w3, #0x96                  	// #150
    34e4:	91008000 	add	x0, x0, #0x20
    34e8:	52800041 	mov	w1, #0x2                   	// #2
    34ec:	94000411 	bl	4530 <spt_ComplainError>
    34f0:	52800040 	mov	w0, #0x2                   	// #2
    34f4:	17ffffef 	b	34b0 <sptMTTKRP_3D+0x1e0>
	sptAssert(nmodes ==3);
    34f8:	12800000 	mov	w0, #0xffffffff            	// #-1
    34fc:	fd0033e8 	str	d8, [sp, #96]
    3500:	97fff730 	bl	11c0 <exit@plt>

0000000000003504 <sptMTTKRP>:
							sptIndex const mode) {
    3504:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    3508:	910003fd 	mov	x29, sp
    350c:	a9025bf5 	stp	x21, x22, [sp, #32]
	sptIndex const nmodes = X->nmodes;
    3510:	b9400016 	ldr	w22, [x0]
	if(nmodes == 3) {
    3514:	71000edf 	cmp	w22, #0x3
    3518:	540015c0 	b.eq	37d0 <sptMTTKRP+0x2cc>  // b.none
	sptNnzIndex const nnz = X->nnz;
    351c:	a90363f7 	stp	x23, x24, [sp, #48]
    3520:	aa0003e5 	mov	x5, x0
	sptIndex const * const ndims = X->ndims;
    3524:	aa0103f7 	mov	x23, x1
    3528:	a9046bf9 	stp	x25, x26, [sp, #64]
    352c:	2a0303e6 	mov	w6, w3
    3530:	aa0203f9 	mov	x25, x2
	sptMatrix * const restrict M = mats[nmodes];
    3534:	f8765827 	ldr	x7, [x1, w22, uxtw #3]
	sptIndex const stride = mats[0]->stride;
    3538:	f9400028 	ldr	x8, [x1]
	sptIndex const * const ndims = X->ndims;
    353c:	f9400804 	ldr	x4, [x0, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    3540:	340003f6 	cbz	w22, 35bc <sptMTTKRP+0xb8>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3544:	b94004e3 	ldr	w3, [x7, #4]
    3548:	aa0803e1 	mov	x1, x8
    354c:	d2800000 	mov	x0, #0x0                   	// #0
    3550:	14000009 	b	3574 <sptMTTKRP+0x70>
		if(mats[i]->nrows != ndims[i]) {
    3554:	b9400022 	ldr	w2, [x1]
    3558:	b8607881 	ldr	w1, [x4, x0, lsl #2]
    355c:	6b01005f 	cmp	w2, w1
    3560:	54001241 	b.ne	37a8 <sptMTTKRP+0x2a4>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3564:	91000400 	add	x0, x0, #0x1
    3568:	6b0002df 	cmp	w22, w0
    356c:	54000289 	b.ls	35bc <sptMTTKRP+0xb8>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3570:	f8607ae1 	ldr	x1, [x23, x0, lsl #3]
    3574:	b9400422 	ldr	w2, [x1, #4]
    3578:	6b03005f 	cmp	w2, w3
    357c:	54fffec0 	b.eq	3554 <sptMTTKRP+0x50>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3580:	b0000004 	adrp	x4, 4000 <sptNewTimer>
    3584:	b0000002 	adrp	x2, 4000 <sptNewTimer>
    3588:	913f4084 	add	x4, x4, #0xfd0
    358c:	913fe042 	add	x2, x2, #0xff8
    3590:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3594:	528007c3 	mov	w3, #0x3e                  	// #62
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3598:	91008000 	add	x0, x0, #0x20
    359c:	52800041 	mov	w1, #0x2                   	// #2
    35a0:	940003e4 	bl	4530 <spt_ComplainError>
    35a4:	a94363f7 	ldp	x23, x24, [sp, #48]
    35a8:	52800040 	mov	w0, #0x2                   	// #2
    35ac:	a9446bf9 	ldp	x25, x26, [sp, #64]
}
    35b0:	a9425bf5 	ldp	x21, x22, [sp, #32]
    35b4:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    35b8:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    35bc:	f8665ae0 	ldr	x0, [x23, w6, uxtw #3]
    35c0:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const stride = mats[0]->stride;
    35c4:	b9400d18 	ldr	w24, [x8, #12]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    35c8:	a9418ca8 	ldp	x8, x3, [x5, #24]
    35cc:	5280031a 	mov	w26, #0x18                  	// #24
	sptIndex const R = mats[mode]->ncols;
    35d0:	29404c02 	ldp	w2, w19, [x0]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    35d4:	52800001 	mov	w1, #0x0                   	// #0
	sptValue * const restrict mvals = M->values;
    35d8:	f94008f5 	ldr	x21, [x7, #16]
    35dc:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    35e0:	9bba0cc3 	umaddl	x3, w6, w26, x3
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    35e4:	1b027f02 	mul	w2, w24, w2
    35e8:	aa1503e0 	mov	x0, x21
	sptNnzIndex const nnz = X->nnz;
    35ec:	a90723e5 	stp	x5, x8, [sp, #112]
	sptNewValueVector(&scratch, R, R);
    35f0:	910223f4 	add	x20, sp, #0x88
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    35f4:	d37ef442 	lsl	x2, x2, #2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    35f8:	f940087c 	ldr	x28, [x3, #16]
	sptValue const * const restrict vals = X->values.data;
    35fc:	f9401cbb 	ldr	x27, [x5, #56]
    3600:	fd0033e8 	str	d8, [sp, #96]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3604:	97fff72b 	bl	12b0 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    3608:	2a1303e2 	mov	w2, w19
    360c:	aa1403e0 	mov	x0, x20
    3610:	aa0203e1 	mov	x1, x2
    3614:	97fffa3b 	bl	1f00 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    3618:	0f000400 	movi	v0.2s, #0x0
    361c:	aa1403e0 	mov	x0, x20
    3620:	97fffa60 	bl	1fa0 <sptConstantValueVector>
	sptNewTimer(&timer, 0);
    3624:	52800001 	mov	w1, #0x0                   	// #0
    3628:	910203e0 	add	x0, sp, #0x80
    362c:	94000275 	bl	4000 <sptNewTimer>
	sptStartTimer(timer);
    3630:	f94043e0 	ldr	x0, [sp, #128]
    3634:	94000293 	bl	4080 <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3638:	a94723e5 	ldp	x5, x8, [sp, #112]
    363c:	b40007c8 	cbz	x8, 3734 <sptMTTKRP+0x230>
		sptIndex times_mat_index = mats_order[1];
    3640:	b9400720 	ldr	w0, [x25, #4]
    3644:	d37ef508 	lsl	x8, x8, #2
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3648:	f94010a7 	ldr	x7, [x5, #32]
			scratch.data[r] = entry * times_mat_values[r];
    364c:	d2800006 	mov	x6, #0x0                   	// #0
    3650:	f9404fe1 	ldr	x1, [sp, #152]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3654:	9bba1c02 	umaddl	x2, w0, w26, x7
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3658:	f8605ae0 	ldr	x0, [x23, w0, uxtw #3]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    365c:	f9400849 	ldr	x9, [x2, #16]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    3660:	f940080a 	ldr	x10, [x0, #16]
    3664:	d503201f 	nop
		sptIndex tmp_mult = tmp_i * stride;
    3668:	b8666920 	ldr	w0, [x9, x6]
		sptValue const entry = vals[x];
    366c:	bc666b61 	ldr	s1, [x27, x6]
		sptIndex tmp_mult = tmp_i * stride;
    3670:	1b007f00 	mul	w0, w24, w0
		for(sptIndex r=0; r<R; ++r) {
    3674:	34000a93 	cbz	w19, 37c4 <sptMTTKRP+0x2c0>
    3678:	8b204942 	add	x2, x10, w0, uxtw #2
    367c:	d2800000 	mov	x0, #0x0                   	// #0
			scratch.data[r] = entry * times_mat_values[r];
    3680:	bc607840 	ldr	s0, [x2, x0, lsl #2]
    3684:	1e210800 	fmul	s0, s0, s1
    3688:	bc207820 	str	s0, [x1, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    368c:	91000400 	add	x0, x0, #0x1
    3690:	6b00027f 	cmp	w19, w0
    3694:	54ffff68 	b.hi	3680 <sptMTTKRP+0x17c>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    3698:	71000adf 	cmp	w22, #0x2
    369c:	54000809 	b.ls	379c <sptMTTKRP+0x298>  // b.plast
		for(sptIndex r=0; r<R; ++r) {
    36a0:	d2800043 	mov	x3, #0x2                   	// #2
    36a4:	d503201f 	nop
			times_mat_index = mats_order[i];
    36a8:	b8637b22 	ldr	w2, [x25, x3, lsl #2]
			for(sptIndex r=0; r<R; ++r) {
    36ac:	d2800000 	mov	x0, #0x0                   	// #0
			tmp_i = times_inds[x];
    36b0:	9bba1c45 	umaddl	x5, w2, w26, x7
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    36b4:	f8625ae4 	ldr	x4, [x23, w2, uxtw #3]
			tmp_i = times_inds[x];
    36b8:	f94008a2 	ldr	x2, [x5, #16]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    36bc:	f9400884 	ldr	x4, [x4, #16]
			tmp_mult = tmp_i * stride;
    36c0:	b8666842 	ldr	w2, [x2, x6]
    36c4:	1b027f02 	mul	w2, w24, w2
			for(sptIndex r=0; r<R; ++r) {
    36c8:	8b224882 	add	x2, x4, w2, uxtw #2
    36cc:	34000113 	cbz	w19, 36ec <sptMTTKRP+0x1e8>
				scratch_data[r] *= times_mat_values[r];
    36d0:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    36d4:	bc607841 	ldr	s1, [x2, x0, lsl #2]
    36d8:	1e210800 	fmul	s0, s0, s1
    36dc:	bc207820 	str	s0, [x1, x0, lsl #2]
			for(sptIndex r=0; r<R; ++r) {
    36e0:	91000400 	add	x0, x0, #0x1
    36e4:	6b00027f 	cmp	w19, w0
    36e8:	54ffff48 	b.hi	36d0 <sptMTTKRP+0x1cc>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    36ec:	91000463 	add	x3, x3, #0x1
    36f0:	6b0302df 	cmp	w22, w3
    36f4:	54fffda8 	b.hi	36a8 <sptMTTKRP+0x1a4>  // b.pmore
		sptIndex tmp_mode = mode_i * stride;
    36f8:	b8666b83 	ldr	w3, [x28, x6]
    36fc:	1b037f03 	mul	w3, w24, w3
		for(sptIndex r=0; r<R; ++r) {
    3700:	34000153 	cbz	w19, 3728 <sptMTTKRP+0x224>
			for(sptIndex r=0; r<R; ++r) {
    3704:	d2800000 	mov	x0, #0x0                   	// #0
			mvals[tmp_mode + r] += scratch_data[r];
    3708:	0b000062 	add	w2, w3, w0
    370c:	bc607821 	ldr	s1, [x1, x0, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3710:	91000400 	add	x0, x0, #0x1
			mvals[tmp_mode + r] += scratch_data[r];
    3714:	bc627aa0 	ldr	s0, [x21, x2, lsl #2]
    3718:	1e212800 	fadd	s0, s0, s1
    371c:	bc227aa0 	str	s0, [x21, x2, lsl #2]
		for(sptIndex r=0; r<R; ++r) {
    3720:	6b00027f 	cmp	w19, w0
    3724:	54ffff28 	b.hi	3708 <sptMTTKRP+0x204>  // b.pmore
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3728:	910010c6 	add	x6, x6, #0x4
    372c:	eb06011f 	cmp	x8, x6
    3730:	54fff9c1 	b.ne	3668 <sptMTTKRP+0x164>  // b.any
	sptStopTimer(timer);
    3734:	f94043e0 	ldr	x0, [sp, #128]
    3738:	94000282 	bl	4140 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    373c:	f94043e0 	ldr	x0, [sp, #128]
    3740:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    3744:	91008021 	add	x1, x1, #0x20
    3748:	940002be 	bl	4240 <sptPrintElapsedTime>
    374c:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3750:	f94043e0 	ldr	x0, [sp, #128]
    3754:	940002f7 	bl	4330 <sptFreeTimer>
	sptFreeValueVector(&scratch);
    3758:	aa1403e0 	mov	x0, x20
    375c:	97fffa82 	bl	2164 <sptFreeValueVector>
	printf("[Total time]: %lf\n", total_time);
    3760:	1e604100 	fmov	d0, d8
    3764:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3768:	91016000 	add	x0, x0, #0x58
    376c:	97fff725 	bl	1400 <printf@plt>
	printf("\n");
    3770:	52800140 	mov	w0, #0xa                   	// #10
    3774:	97fff72f 	bl	1430 <putchar@plt>
	return 0;
    3778:	52800000 	mov	w0, #0x0                   	// #0
    377c:	fd4033e8 	ldr	d8, [sp, #96]
    3780:	a94153f3 	ldp	x19, x20, [sp, #16]
}
    3784:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    3788:	a94363f7 	ldp	x23, x24, [sp, #48]
    378c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3790:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3794:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3798:	d65f03c0 	ret
		sptIndex tmp_mode = mode_i * stride;
    379c:	b8666b83 	ldr	w3, [x28, x6]
    37a0:	1b037f03 	mul	w3, w24, w3
		for(sptIndex r=0; r<R; ++r) {
    37a4:	17ffffd8 	b	3704 <sptMTTKRP+0x200>
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    37a8:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    37ac:	b0000002 	adrp	x2, 4000 <sptNewTimer>
    37b0:	9100e084 	add	x4, x4, #0x38
    37b4:	913fe042 	add	x2, x2, #0xff8
    37b8:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    37bc:	52800823 	mov	w3, #0x41                  	// #65
    37c0:	17ffff76 	b	3598 <sptMTTKRP+0x94>
		for(sptIndex i=2; i<nmodes; ++i) {
    37c4:	71000adf 	cmp	w22, #0x2
    37c8:	54fff6c8 	b.hi	36a0 <sptMTTKRP+0x19c>  // b.pmore
    37cc:	17ffffd7 	b	3728 <sptMTTKRP+0x224>
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    37d0:	97fffec0 	bl	32d0 <sptMTTKRP_3D>
    37d4:	34ffeee0 	cbz	w0, 35b0 <sptMTTKRP+0xac>
    37d8:	12800000 	mov	w0, #0xffffffff            	// #-1
    37dc:	a90153f3 	stp	x19, x20, [sp, #16]
    37e0:	a90363f7 	stp	x23, x24, [sp, #48]
    37e4:	a9046bf9 	stp	x25, x26, [sp, #64]
    37e8:	a90573fb 	stp	x27, x28, [sp, #80]
    37ec:	fd0033e8 	str	d8, [sp, #96]
    37f0:	97fff674 	bl	11c0 <exit@plt>
    37f4:	d503201f 	nop
    37f8:	d503201f 	nop
    37fc:	d503201f 	nop

0000000000003800 <sptOmpMTTKRP._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3800:	a9b27bfd 	stp	x29, x30, [sp, #-224]!
    3804:	910003fd 	mov	x29, sp
    3808:	a90153f3 	stp	x19, x20, [sp, #16]
    380c:	f9400c14 	ldr	x20, [x0, #24]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3810:	b4001114 	cbz	x20, 3a30 <sptOmpMTTKRP._omp_fn.0+0x230>
    3814:	a90363f7 	stp	x23, x24, [sp, #48]
    3818:	aa0003f8 	mov	x24, x0
    381c:	a9046bf9 	stp	x25, x26, [sp, #64]
    3820:	97fff6b8 	bl	1300 <omp_get_num_threads@plt>
    3824:	93407c13 	sxtw	x19, w0
    3828:	97fff67e 	bl	1220 <omp_get_thread_num@plt>
    382c:	93407c1a 	sxtw	x26, w0
    3830:	9ad30a81 	udiv	x1, x20, x19
    3834:	9b13d020 	msub	x0, x1, x19, x20
    3838:	eb00035f 	cmp	x26, x0
    383c:	540011c3 	b.cc	3a74 <sptOmpMTTKRP._omp_fn.0+0x274>  // b.lo, b.ul, b.last
    3840:	9b1a0020 	madd	x0, x1, x26, x0
    3844:	f90043e0 	str	x0, [sp, #128]
    3848:	8b000021 	add	x1, x1, x0
    384c:	f9004be1 	str	x1, [sp, #144]
    3850:	eb01001f 	cmp	x0, x1
    3854:	54000f42 	b.cs	3a3c <sptOmpMTTKRP._omp_fn.0+0x23c>  // b.hs, b.nlast
    3858:	a90573fb 	stp	x27, x28, [sp, #80]
    385c:	910323e0 	add	x0, sp, #0xc8
#pragma omp parallel for schedule(static) num_threads(tk)
    3860:	b940431c 	ldr	w28, [x24, #64]
    3864:	f90047e0 	str	x0, [sp, #136]
		sptNewValueVector(&scratch, R, R);
		sptConstantValueVector(&scratch, 0);

		sptIndex times_mat_index = mats_order[1];
		sptMatrix * times_mat = mats[times_mat_index];
		sptIndex * times_inds = X->inds[times_mat_index].data;
    3868:	52800314 	mov	w20, #0x18                  	// #24
    386c:	51000799 	sub	w25, w28, #0x1
#pragma omp parallel for schedule(static) num_threads(tk)
    3870:	b9403b17 	ldr	w23, [x24, #56]
    3874:	91000739 	add	x25, x25, #0x1
    3878:	a9025bf5 	stp	x21, x22, [sp, #32]
    387c:	f9400300 	ldr	x0, [x24]
    3880:	f90053e0 	str	x0, [sp, #160]
    3884:	d37ef720 	lsl	x0, x25, #2
    3888:	f9003fe0 	str	x0, [sp, #120]
		sptNewValueVector(&scratch, R, R);
    388c:	2a1c03e0 	mov	w0, w28
    3890:	f90057e0 	str	x0, [sp, #168]
#pragma omp parallel for schedule(static) num_threads(tk)
    3894:	f9401300 	ldr	x0, [x24, #32]
    3898:	f9004fe0 	str	x0, [sp, #152]
    389c:	f9401700 	ldr	x0, [x24, #40]
    38a0:	f9005be0 	str	x0, [sp, #176]
    38a4:	a940d716 	ldp	x22, x21, [x24, #8]
    38a8:	fd0033e8 	str	d8, [sp, #96]
    38ac:	f9401b00 	ldr	x0, [x24, #48]
    38b0:	f9005fe0 	str	x0, [sp, #184]
    38b4:	b9403f18 	ldr	w24, [x24, #60]
		sptNewValueVector(&scratch, R, R);
    38b8:	f94057e1 	ldr	x1, [sp, #168]
    38bc:	f94047f3 	ldr	x19, [sp, #136]
    38c0:	aa0103e2 	mov	x2, x1
    38c4:	aa1303e0 	mov	x0, x19
    38c8:	97fff98e 	bl	1f00 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    38cc:	0f000400 	movi	v0.2s, #0x0
    38d0:	aa1303e0 	mov	x0, x19
    38d4:	97fff9b3 	bl	1fa0 <sptConstantValueVector>
		sptIndex * times_inds = X->inds[times_mat_index].data;
    38d8:	a94987e2 	ldp	x2, x1, [sp, #152]
		sptIndex times_mat_index = mats_order[1];
    38dc:	b94006a0 	ldr	w0, [x21, #4]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    38e0:	f9401027 	ldr	x7, [x1, #32]
    38e4:	f94043e1 	ldr	x1, [sp, #128]
		sptIndex tmp_i = times_inds[x];
		sptValue const entry = vals[x];
    38e8:	bc617841 	ldr	s1, [x2, x1, lsl #2]
    38ec:	d37ef428 	lsl	x8, x1, #2
		sptMatrix * times_mat = mats[times_mat_index];
    38f0:	f8605ac2 	ldr	x2, [x22, w0, uxtw #3]
		sptIndex tmp_i = times_inds[x];
    38f4:	9bb41c00 	umaddl	x0, w0, w20, x7
    38f8:	f9400800 	ldr	x0, [x0, #16]
    38fc:	b8617803 	ldr	w3, [x0, x1, lsl #2]
#pragma omp simd
		for(sptIndex r=0; r<R; ++r) {
    3900:	34000b3c 	cbz	w28, 3a64 <sptOmpMTTKRP._omp_fn.0+0x264>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    3904:	1b037f03 	mul	w3, w24, w3
    3908:	d2800000 	mov	x0, #0x0                   	// #0
    390c:	f9400845 	ldr	x5, [x2, #16]
    3910:	f9406fe1 	ldr	x1, [sp, #216]
    3914:	d503201f 	nop
    3918:	0b000062 	add	w2, w3, w0
    391c:	bc6278a0 	ldr	s0, [x5, x2, lsl #2]
    3920:	1e210800 	fmul	s0, s0, s1
    3924:	bc207820 	str	s0, [x1, x0, lsl #2]
    3928:	91000400 	add	x0, x0, #0x1
    392c:	6b00039f 	cmp	w28, w0
    3930:	54ffff48 	b.hi	3918 <sptOmpMTTKRP._omp_fn.0+0x118>  // b.pmore
		}

		for(sptIndex i=2; i<nmodes; ++i) {
    3934:	71000aff 	cmp	w23, #0x2
    3938:	540008c9 	b.ls	3a50 <sptOmpMTTKRP._omp_fn.0+0x250>  // b.plast
			times_inds = X->inds[times_mat_index].data;
			tmp_i = times_inds[x];

#pragma omp simd
			for(sptIndex r=0; r<R; ++r) {
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    393c:	d2800046 	mov	x6, #0x2                   	// #2
			times_mat_index = mats_order[i];
    3940:	b8667aa0 	ldr	w0, [x21, x6, lsl #2]
			tmp_i = times_inds[x];
    3944:	9bb41c02 	umaddl	x2, w0, w20, x7
			times_mat = mats[times_mat_index];
    3948:	f8605ac5 	ldr	x5, [x22, w0, uxtw #3]
			tmp_i = times_inds[x];
    394c:	f9400840 	ldr	x0, [x2, #16]
    3950:	b8686803 	ldr	w3, [x0, x8]
			for(sptIndex r=0; r<R; ++r) {
    3954:	340001bc 	cbz	w28, 3988 <sptOmpMTTKRP._omp_fn.0+0x188>
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    3958:	1b037f03 	mul	w3, w24, w3
    395c:	d2800000 	mov	x0, #0x0                   	// #0
    3960:	f94008a5 	ldr	x5, [x5, #16]
    3964:	d503201f 	nop
    3968:	0b000062 	add	w2, w3, w0
    396c:	bc607820 	ldr	s0, [x1, x0, lsl #2]
    3970:	bc6278a1 	ldr	s1, [x5, x2, lsl #2]
    3974:	1e210800 	fmul	s0, s0, s1
    3978:	bc207820 	str	s0, [x1, x0, lsl #2]
    397c:	91000400 	add	x0, x0, #0x1
    3980:	6b00039f 	cmp	w28, w0
    3984:	54ffff28 	b.hi	3968 <sptOmpMTTKRP._omp_fn.0+0x168>  // b.pmore
		for(sptIndex i=2; i<nmodes; ++i) {
    3988:	910004c6 	add	x6, x6, #0x1
    398c:	6b0602ff 	cmp	w23, w6
    3990:	54fffd88 	b.hi	3940 <sptOmpMTTKRP._omp_fn.0+0x140>  // b.pmore
			}
		}

		sptIndex const mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3994:	f94043e2 	ldr	x2, [sp, #128]
    3998:	f9405be0 	ldr	x0, [sp, #176]
    399c:	b8627813 	ldr	w19, [x0, x2, lsl #2]
    39a0:	1b137f13 	mul	w19, w24, w19
		for(sptIndex r=0; r<R; ++r) {
    39a4:	340002bc 	cbz	w28, 39f8 <sptOmpMTTKRP._omp_fn.0+0x1f8>
    39a8:	f9405fe0 	ldr	x0, [sp, #184]
#pragma omp parallel for schedule(static) num_threads(tk)
    39ac:	d280001b 	mov	x27, #0x0                   	// #0
    39b0:	8b130813 	add	x19, x0, x19, lsl #2
    39b4:	14000002 	b	39bc <sptOmpMTTKRP._omp_fn.0+0x1bc>
    39b8:	f9406fe1 	ldr	x1, [sp, #216]
    39bc:	8b13037a 	add	x26, x27, x19
#pragma omp atomic update
			mvals_row[r] += scratch.data[r];
    39c0:	bc7b6828 	ldr	s8, [x1, x27]
    39c4:	b9400359 	ldr	w25, [x26]
    39c8:	1e270320 	fmov	s0, w25
    39cc:	aa1a03e2 	mov	x2, x26
    39d0:	2a1903e0 	mov	w0, w25
    39d4:	1e202900 	fadd	s0, s8, s0
    39d8:	1e260001 	fmov	w1, s0
    39dc:	9400035d 	bl	4750 <__aarch64_cas4_acq_rel>
    39e0:	6b00033f 	cmp	w25, w0
    39e4:	540004e1 	b.ne	3a80 <sptOmpMTTKRP._omp_fn.0+0x280>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    39e8:	f9403fe0 	ldr	x0, [sp, #120]
    39ec:	9100137b 	add	x27, x27, #0x4
    39f0:	eb00037f 	cmp	x27, x0
    39f4:	54fffe21 	b.ne	39b8 <sptOmpMTTKRP._omp_fn.0+0x1b8>  // b.any
		}

		sptFreeValueVector(&scratch);
    39f8:	f94043e0 	ldr	x0, [sp, #128]
    39fc:	91000400 	add	x0, x0, #0x1
    3a00:	aa0003f3 	mov	x19, x0
    3a04:	f90043f3 	str	x19, [sp, #128]
    3a08:	f94047e0 	ldr	x0, [sp, #136]
    3a0c:	97fff9d6 	bl	2164 <sptFreeValueVector>
    3a10:	f9404be1 	ldr	x1, [sp, #144]
    3a14:	eb13003f 	cmp	x1, x19
    3a18:	54fff508 	b.hi	38b8 <sptOmpMTTKRP._omp_fn.0+0xb8>  // b.pmore
    3a1c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3a20:	a94363f7 	ldp	x23, x24, [sp, #48]
    3a24:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3a28:	a94573fb 	ldp	x27, x28, [sp, #80]
    3a2c:	fd4033e8 	ldr	d8, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3a30:	a94153f3 	ldp	x19, x20, [sp, #16]
    3a34:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    3a38:	d65f03c0 	ret
    3a3c:	a94153f3 	ldp	x19, x20, [sp, #16]
    3a40:	a94363f7 	ldp	x23, x24, [sp, #48]
    3a44:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3a48:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    3a4c:	d65f03c0 	ret
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3a50:	f94043e2 	ldr	x2, [sp, #128]
    3a54:	f9405be0 	ldr	x0, [sp, #176]
    3a58:	b8627813 	ldr	w19, [x0, x2, lsl #2]
    3a5c:	1b137f13 	mul	w19, w24, w19
		for(sptIndex r=0; r<R; ++r) {
    3a60:	17ffffd2 	b	39a8 <sptOmpMTTKRP._omp_fn.0+0x1a8>
		for(sptIndex i=2; i<nmodes; ++i) {
    3a64:	71000aff 	cmp	w23, #0x2
    3a68:	54fffc89 	b.ls	39f8 <sptOmpMTTKRP._omp_fn.0+0x1f8>  // b.plast
    3a6c:	f9406fe1 	ldr	x1, [sp, #216]
    3a70:	17ffffb3 	b	393c <sptOmpMTTKRP._omp_fn.0+0x13c>
    3a74:	91000421 	add	x1, x1, #0x1
#pragma omp parallel for schedule(static) num_threads(tk)
    3a78:	d2800000 	mov	x0, #0x0                   	// #0
    3a7c:	17ffff71 	b	3840 <sptOmpMTTKRP._omp_fn.0+0x40>
			mvals_row[r] += scratch.data[r];
    3a80:	2a0003f9 	mov	w25, w0
    3a84:	17ffffd1 	b	39c8 <sptOmpMTTKRP._omp_fn.0+0x1c8>
    3a88:	d503201f 	nop
    3a8c:	d503201f 	nop

0000000000003a90 <sptOmpMTTKRP_3D._omp_fn.0>:
	sptTimer timer;
	sptNewTimer(&timer, 0);
	double comp_time, total_time;

	sptStartTimer(timer);
#pragma omp parallel for schedule(static) num_threads(tk)
    3a90:	a9b67bfd 	stp	x29, x30, [sp, #-160]!
    3a94:	910003fd 	mov	x29, sp
    3a98:	a9025bf5 	stp	x21, x22, [sp, #32]
    3a9c:	f9400016 	ldr	x22, [x0]
	for(sptNnzIndex x=0; x<nnz; ++x) {
    3aa0:	b4000a36 	cbz	x22, 3be4 <sptOmpMTTKRP_3D._omp_fn.0+0x154>
    3aa4:	a90153f3 	stp	x19, x20, [sp, #16]
    3aa8:	aa0003f3 	mov	x19, x0
    3aac:	a90363f7 	stp	x23, x24, [sp, #48]
    3ab0:	97fff614 	bl	1300 <omp_get_num_threads@plt>
    3ab4:	93407c14 	sxtw	x20, w0
    3ab8:	97fff5da 	bl	1220 <omp_get_thread_num@plt>
    3abc:	93407c15 	sxtw	x21, w0
    3ac0:	9ad40ad7 	udiv	x23, x22, x20
    3ac4:	9b14dae0 	msub	x0, x23, x20, x22
    3ac8:	eb0002bf 	cmp	x21, x0
    3acc:	540009c3 	b.cc	3c04 <sptOmpMTTKRP_3D._omp_fn.0+0x174>  // b.lo, b.ul, b.last
    3ad0:	9b1502f5 	madd	x21, x23, x21, x0
    3ad4:	8b1502e0 	add	x0, x23, x21
    3ad8:	f9003fe0 	str	x0, [sp, #120]
    3adc:	eb0002bf 	cmp	x21, x0
    3ae0:	54000882 	b.cs	3bf0 <sptOmpMTTKRP_3D._omp_fn.0+0x160>  // b.hs, b.nlast
    3ae4:	a90573fb 	stp	x27, x28, [sp, #80]
#pragma omp parallel for schedule(static) num_threads(tk)
    3ae8:	b9404276 	ldr	w22, [x19, #64]
    3aec:	a940ee60 	ldp	x0, x27, [x19, #8]
    3af0:	f9004be0 	str	x0, [sp, #144]
    3af4:	f9400e60 	ldr	x0, [x19, #24]
    3af8:	f9004fe0 	str	x0, [sp, #152]
    3afc:	f9401660 	ldr	x0, [x19, #40]
    3b00:	a9046bf9 	stp	x25, x26, [sp, #64]
    3b04:	f90047e0 	str	x0, [sp, #136]
    3b08:	f940127a 	ldr	x26, [x19, #32]
    3b0c:	6d0627e8 	stp	d8, d9, [sp, #96]
    3b10:	f9401a79 	ldr	x25, [x19, #48]
    3b14:	f9401e60 	ldr	x0, [x19, #56]
    3b18:	f90043e0 	str	x0, [sp, #128]
    3b1c:	b9404660 	ldr	w0, [x19, #68]
    3b20:	b90077e0 	str	w0, [sp, #116]
    3b24:	d503201f 	nop
		sptIndex mode_i = mode_ind[x];
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
		sptIndex tmp_i_1 = times_inds_1[x];
    3b28:	f94047e0 	ldr	x0, [sp, #136]
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3b2c:	b8757b62 	ldr	w2, [x27, x21, lsl #2]
		sptIndex tmp_i_1 = times_inds_1[x];
    3b30:	b8757814 	ldr	w20, [x0, x21, lsl #2]
		sptIndex tmp_i_2 = times_inds_2[x];
    3b34:	f94043e0 	ldr	x0, [sp, #128]
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    3b38:	1b027ec2 	mul	w2, w22, w2
		sptIndex tmp_i_2 = times_inds_2[x];
    3b3c:	b8757813 	ldr	w19, [x0, x21, lsl #2]
		sptValue entry = vals[x];
    3b40:	f9404be0 	ldr	x0, [sp, #144]
    3b44:	bc757809 	ldr	s9, [x0, x21, lsl #2]

		for(sptIndex r=0; r<R; ++r) {
    3b48:	b94077e0 	ldr	w0, [sp, #116]
    3b4c:	340003a0 	cbz	w0, 3bc0 <sptOmpMTTKRP_3D._omp_fn.0+0x130>
    3b50:	f9404fe0 	ldr	x0, [sp, #152]
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3b54:	1b137ed3 	mul	w19, w22, w19
    3b58:	1b147ed4 	mul	w20, w22, w20
    3b5c:	2a1303fc 	mov	w28, w19
    3b60:	8b224817 	add	x23, x0, w2, uxtw #2
    3b64:	b94077e0 	ldr	w0, [sp, #116]
    3b68:	4b130294 	sub	w20, w20, w19
    3b6c:	0b130013 	add	w19, w0, w19
    3b70:	f9400b44 	ldr	x4, [x26, #16]
    3b74:	0b1c0280 	add	w0, w20, w28
    3b78:	f9400b21 	ldr	x1, [x25, #16]
    3b7c:	bc607888 	ldr	s8, [x4, x0, lsl #2]
    3b80:	bc7c5820 	ldr	s0, [x1, w28, uxtw #2]
    3b84:	1e280928 	fmul	s8, s9, s8
    3b88:	1e200908 	fmul	s8, s8, s0
    3b8c:	b94002f8 	ldr	w24, [x23]
    3b90:	1e270300 	fmov	s0, w24
    3b94:	aa1703e2 	mov	x2, x23
    3b98:	2a1803e0 	mov	w0, w24
    3b9c:	1e202900 	fadd	s0, s8, s0
    3ba0:	1e260001 	fmov	w1, s0
    3ba4:	940002eb 	bl	4750 <__aarch64_cas4_acq_rel>
    3ba8:	6b00031f 	cmp	w24, w0
    3bac:	54000321 	b.ne	3c10 <sptOmpMTTKRP_3D._omp_fn.0+0x180>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    3bb0:	1100079c 	add	w28, w28, #0x1
    3bb4:	910012f7 	add	x23, x23, #0x4
    3bb8:	6b13039f 	cmp	w28, w19
    3bbc:	54fffda1 	b.ne	3b70 <sptOmpMTTKRP_3D._omp_fn.0+0xe0>  // b.any
    3bc0:	f9403fe0 	ldr	x0, [sp, #120]
    3bc4:	910006b5 	add	x21, x21, #0x1
    3bc8:	eb15001f 	cmp	x0, x21
    3bcc:	54fffae1 	b.ne	3b28 <sptOmpMTTKRP_3D._omp_fn.0+0x98>  // b.any
    3bd0:	a94153f3 	ldp	x19, x20, [sp, #16]
    3bd4:	a94363f7 	ldp	x23, x24, [sp, #48]
    3bd8:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3bdc:	a94573fb 	ldp	x27, x28, [sp, #80]
    3be0:	6d4627e8 	ldp	d8, d9, [sp, #96]
#pragma omp parallel for schedule(static) num_threads(tk)
    3be4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3be8:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3bec:	d65f03c0 	ret
    3bf0:	a94153f3 	ldp	x19, x20, [sp, #16]
    3bf4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3bf8:	a94363f7 	ldp	x23, x24, [sp, #48]
    3bfc:	a8ca7bfd 	ldp	x29, x30, [sp], #160
    3c00:	d65f03c0 	ret
    3c04:	910006f7 	add	x23, x23, #0x1
    3c08:	d2800000 	mov	x0, #0x0                   	// #0
    3c0c:	17ffffb1 	b	3ad0 <sptOmpMTTKRP_3D._omp_fn.0+0x40>
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    3c10:	2a0003f8 	mov	w24, w0
    3c14:	17ffffdf 	b	3b90 <sptOmpMTTKRP_3D._omp_fn.0+0x100>
    3c18:	d503201f 	nop
    3c1c:	d503201f 	nop

0000000000003c20 <sptOmpMTTKRP_3D>:
{
    3c20:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    3c24:	910003fd 	mov	x29, sp
    3c28:	a90153f3 	stp	x19, x20, [sp, #16]
    3c2c:	aa0003f4 	mov	x20, x0
    3c30:	aa0103f3 	mov	x19, x1
	sptIndex const stride = mats[0]->stride;
    3c34:	f9400020 	ldr	x0, [x1]
{
    3c38:	a90363f7 	stp	x23, x24, [sp, #48]
	sptAssert(nmodes ==3);
    3c3c:	b9400281 	ldr	w1, [x20]
{
    3c40:	2a0403f7 	mov	w23, w4
    3c44:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const stride = mats[0]->stride;
    3c48:	b9400c18 	ldr	w24, [x0, #12]
	sptNnzIndex const nnz = X->nnz;
    3c4c:	a9416a84 	ldp	x4, x26, [x20, #16]
	sptValue const * const restrict vals = X->values.data;
    3c50:	f9401e99 	ldr	x25, [x20, #56]
	sptAssert(nmodes ==3);
    3c54:	a9025bf5 	stp	x21, x22, [sp, #32]
    3c58:	71000c3f 	cmp	w1, #0x3
    3c5c:	54000d41 	b.ne	3e04 <sptOmpMTTKRP_3D+0x1e4>  // b.any
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3c60:	f9400e66 	ldr	x6, [x19, #24]
    3c64:	aa0203f6 	mov	x22, x2
    3c68:	b9400401 	ldr	w1, [x0, #4]
    3c6c:	d2800005 	mov	x5, #0x0                   	// #0
    3c70:	b94004c2 	ldr	w2, [x6, #4]
    3c74:	6b02003f 	cmp	w1, w2
    3c78:	54000181 	b.ne	3ca8 <sptOmpMTTKRP_3D+0x88>  // b.any
		if(mats[i]->nrows != ndims[i]) {
    3c7c:	b9400001 	ldr	w1, [x0]
    3c80:	b8657880 	ldr	w0, [x4, x5, lsl #2]
    3c84:	6b00003f 	cmp	w1, w0
    3c88:	54000a81 	b.ne	3dd8 <sptOmpMTTKRP_3D+0x1b8>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3c8c:	910004a5 	add	x5, x5, #0x1
    3c90:	f1000cbf 	cmp	x5, #0x3
    3c94:	54000200 	b.eq	3cd4 <sptOmpMTTKRP_3D+0xb4>  // b.none
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3c98:	f8657a60 	ldr	x0, [x19, x5, lsl #3]
    3c9c:	b9400401 	ldr	w1, [x0, #4]
    3ca0:	6b02003f 	cmp	w1, w2
    3ca4:	54fffec0 	b.eq	3c7c <sptOmpMTTKRP_3D+0x5c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3ca8:	b0000004 	adrp	x4, 4000 <sptNewTimer>
    3cac:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    3cb0:	913f4084 	add	x4, x4, #0xfd0
    3cb4:	9101c042 	add	x2, x2, #0x70
    3cb8:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3cbc:	528012c3 	mov	w3, #0x96                  	// #150
    3cc0:	91026000 	add	x0, x0, #0x98
    3cc4:	52800041 	mov	w1, #0x2                   	// #2
    3cc8:	9400021a 	bl	4530 <spt_ComplainError>
    3ccc:	52800040 	mov	w0, #0x2                   	// #2
    3cd0:	1400003c 	b	3dc0 <sptOmpMTTKRP_3D+0x1a0>
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3cd4:	f9401284 	ldr	x4, [x20, #32]
    3cd8:	52800315 	mov	w21, #0x18                  	// #24
	sptIndex const tmpI = mats[mode]->nrows;
    3cdc:	f8635a60 	ldr	x0, [x19, w3, uxtw #3]
	sptIndex const R = mats[mode]->ncols;
    3ce0:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3ce4:	9bb51063 	umaddl	x3, w3, w21, x4
	sptValue * const restrict mvals = mats[nmodes]->values;
    3ce8:	f94008db 	ldr	x27, [x6, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3cec:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    3cf0:	29407002 	ldp	w2, w28, [x0]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3cf4:	aa1b03e0 	mov	x0, x27
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3cf8:	f9400866 	ldr	x6, [x3, #16]
    3cfc:	fd0033e8 	str	d8, [sp, #96]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d00:	1b027f02 	mul	w2, w24, w2
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    3d04:	f9003fe6 	str	x6, [sp, #120]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3d08:	d37ef442 	lsl	x2, x2, #2
    3d0c:	97fff569 	bl	12b0 <memset@plt>
	sptIndex times_mat_index_2 = mats_order[2];
    3d10:	29408ac3 	ldp	w3, w2, [x22, #4]
	sptNewTimer(&timer, 0);
    3d14:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3d18:	f9401284 	ldr	x4, [x20, #32]
	sptNewTimer(&timer, 0);
    3d1c:	910203e0 	add	x0, sp, #0x80
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    3d20:	f8635a65 	ldr	x5, [x19, w3, uxtw #3]
    3d24:	f9003be5 	str	x5, [sp, #112]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3d28:	9bb51063 	umaddl	x3, w3, w21, x4
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3d2c:	9bb51055 	umaddl	x21, w2, w21, x4
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    3d30:	f8625a74 	ldr	x20, [x19, w2, uxtw #3]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    3d34:	f9400876 	ldr	x22, [x3, #16]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    3d38:	f9400ab3 	ldr	x19, [x21, #16]
	sptNewTimer(&timer, 0);
    3d3c:	940000b1 	bl	4000 <sptNewTimer>
	sptStartTimer(timer);
    3d40:	f94043e0 	ldr	x0, [sp, #128]
    3d44:	940000cf 	bl	4080 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    3d48:	a908e7fa 	stp	x26, x25, [sp, #136]
    3d4c:	2a1703e2 	mov	w2, w23
    3d50:	a9471be5 	ldp	x5, x6, [sp, #112]
    3d54:	52800003 	mov	w3, #0x0                   	// #0
    3d58:	910223e1 	add	x1, sp, #0x88
    3d5c:	90000000 	adrp	x0, 3000 <sptRandomizeMatrix+0xc0>
    3d60:	912a4000 	add	x0, x0, #0xa90
    3d64:	a909efe6 	stp	x6, x27, [sp, #152]
    3d68:	a90adbe5 	stp	x5, x22, [sp, #168]
    3d6c:	a90bcff4 	stp	x20, x19, [sp, #184]
    3d70:	291973f8 	stp	w24, w28, [sp, #200]
    3d74:	97fff5ab 	bl	1420 <GOMP_parallel@plt>
		}
	}
	sptStopTimer(timer);
    3d78:	f94043e0 	ldr	x0, [sp, #128]
    3d7c:	940000f1 	bl	4140 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    3d80:	f94043e0 	ldr	x0, [sp, #128]
    3d84:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    3d88:	91026021 	add	x1, x1, #0x98
    3d8c:	9400012d 	bl	4240 <sptPrintElapsedTime>
    3d90:	1e604008 	fmov	d8, d0

	sptFreeTimer(timer);
    3d94:	f94043e0 	ldr	x0, [sp, #128]
    3d98:	94000166 	bl	4330 <sptFreeTimer>

	total_time = comp_time;
	printf("[Total time]: %lf\n", total_time);
    3d9c:	1e604100 	fmov	d0, d8
    3da0:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3da4:	91016000 	add	x0, x0, #0x58
    3da8:	97fff596 	bl	1400 <printf@plt>
	printf("\n");
    3dac:	52800140 	mov	w0, #0xa                   	// #10
    3db0:	97fff5a0 	bl	1430 <putchar@plt>

	return 0;
    3db4:	a94573fb 	ldp	x27, x28, [sp, #80]
    3db8:	52800000 	mov	w0, #0x0                   	// #0
    3dbc:	fd4033e8 	ldr	d8, [sp, #96]
}
    3dc0:	a94153f3 	ldp	x19, x20, [sp, #16]
    3dc4:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3dc8:	a94363f7 	ldp	x23, x24, [sp, #48]
    3dcc:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3dd0:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3dd4:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3dd8:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    3ddc:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    3de0:	9100e084 	add	x4, x4, #0x38
    3de4:	9101c042 	add	x2, x2, #0x70
    3de8:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3dec:	52801323 	mov	w3, #0x99                  	// #153
    3df0:	91026000 	add	x0, x0, #0x98
    3df4:	52800041 	mov	w1, #0x2                   	// #2
    3df8:	940001ce 	bl	4530 <spt_ComplainError>
    3dfc:	52800040 	mov	w0, #0x2                   	// #2
    3e00:	17fffff0 	b	3dc0 <sptOmpMTTKRP_3D+0x1a0>
	sptAssert(nmodes ==3);
    3e04:	12800000 	mov	w0, #0xffffffff            	// #-1
    3e08:	a90573fb 	stp	x27, x28, [sp, #80]
    3e0c:	fd0033e8 	str	d8, [sp, #96]
    3e10:	97fff4ec 	bl	11c0 <exit@plt>

0000000000003e14 <sptOmpMTTKRP>:
{
    3e14:	a9b37bfd 	stp	x29, x30, [sp, #-208]!
    3e18:	910003fd 	mov	x29, sp
    3e1c:	a90153f3 	stp	x19, x20, [sp, #16]
	sptIndex const nmodes = X->nmodes;
    3e20:	b9400014 	ldr	w20, [x0]
	if(nmodes == 3) {
    3e24:	71000e9f 	cmp	w20, #0x3
    3e28:	54000da0 	b.eq	3fdc <sptOmpMTTKRP+0x1c8>  // b.none
	sptNnzIndex const nnz = X->nnz;
    3e2c:	a90363f7 	stp	x23, x24, [sp, #48]
    3e30:	aa0003f7 	mov	x23, x0
	sptIndex const * const ndims = X->ndims;
    3e34:	aa0103f3 	mov	x19, x1
    3e38:	a9025bf5 	stp	x21, x22, [sp, #32]
    3e3c:	2a0303e9 	mov	w9, w3
    3e40:	aa0203f6 	mov	x22, x2
    3e44:	2a0403f5 	mov	w21, w4
	sptValue * const restrict mvals = mats[nmodes]->values;
    3e48:	f8745820 	ldr	x0, [x1, w20, uxtw #3]
	sptIndex const stride = mats[0]->stride;
    3e4c:	f9400022 	ldr	x2, [x1]
	sptIndex const * const ndims = X->ndims;
    3e50:	f9400ae8 	ldr	x8, [x23, #16]
	for(sptIndex i=0; i<nmodes; ++i) {
    3e54:	340003f4 	cbz	w20, 3ed0 <sptOmpMTTKRP+0xbc>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3e58:	b9400401 	ldr	w1, [x0, #4]
    3e5c:	aa0203e6 	mov	x6, x2
    3e60:	d2800005 	mov	x5, #0x0                   	// #0
    3e64:	14000009 	b	3e88 <sptOmpMTTKRP+0x74>
		if(mats[i]->nrows != ndims[i]) {
    3e68:	b94000c7 	ldr	w7, [x6]
    3e6c:	b8657906 	ldr	w6, [x8, x5, lsl #2]
    3e70:	6b0600ff 	cmp	w7, w6
    3e74:	54000a61 	b.ne	3fc0 <sptOmpMTTKRP+0x1ac>  // b.any
	for(sptIndex i=0; i<nmodes; ++i) {
    3e78:	910004a5 	add	x5, x5, #0x1
    3e7c:	6b05029f 	cmp	w20, w5
    3e80:	54000289 	b.ls	3ed0 <sptOmpMTTKRP+0xbc>  // b.plast
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    3e84:	f8657a66 	ldr	x6, [x19, x5, lsl #3]
    3e88:	b94004c7 	ldr	w7, [x6, #4]
    3e8c:	6b0100ff 	cmp	w7, w1
    3e90:	54fffec0 	b.eq	3e68 <sptOmpMTTKRP+0x54>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    3e94:	b0000004 	adrp	x4, 4000 <sptNewTimer>
    3e98:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    3e9c:	913f4084 	add	x4, x4, #0xfd0
    3ea0:	9101c042 	add	x2, x2, #0x70
    3ea4:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3ea8:	52800823 	mov	w3, #0x41                  	// #65
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3eac:	91026000 	add	x0, x0, #0x98
    3eb0:	52800041 	mov	w1, #0x2                   	// #2
    3eb4:	9400019f 	bl	4530 <spt_ComplainError>
    3eb8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3ebc:	52800040 	mov	w0, #0x2                   	// #2
    3ec0:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    3ec4:	a94153f3 	ldp	x19, x20, [sp, #16]
    3ec8:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3ecc:	d65f03c0 	ret
	sptIndex const tmpI = mats[mode]->nrows;
    3ed0:	f8695a64 	ldr	x4, [x19, w9, uxtw #3]
    3ed4:	a9046bf9 	stp	x25, x26, [sp, #64]
	sptIndex const stride = mats[0]->stride;
    3ed8:	b9400c58 	ldr	w24, [x2, #12]
	sptIndex const * const mode_ind = X->inds[mode].data;
    3edc:	52800305 	mov	w5, #0x18                  	// #24
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3ee0:	52800001 	mov	w1, #0x0                   	// #0
	sptIndex const R = mats[mode]->ncols;
    3ee4:	29406882 	ldp	w2, w26, [x4]
	sptIndex const * const mode_ind = X->inds[mode].data;
    3ee8:	a90573fb 	stp	x27, x28, [sp, #80]
    3eec:	a9418ee4 	ldp	x4, x3, [x23, #24]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3ef0:	1b027f02 	mul	w2, w24, w2
	sptValue * const restrict mvals = mats[nmodes]->values;
    3ef4:	f9400819 	ldr	x25, [x0, #16]
	sptNnzIndex const nnz = X->nnz;
    3ef8:	f9003fe4 	str	x4, [sp, #120]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3efc:	d37ef442 	lsl	x2, x2, #2
    3f00:	aa1903e0 	mov	x0, x25
	sptIndex const * const mode_ind = X->inds[mode].data;
    3f04:	9ba50d23 	umaddl	x3, w9, w5, x3
	sptValue const * const vals = X->values.data;
    3f08:	f9401efc 	ldr	x28, [x23, #56]
	sptIndex const * const mode_ind = X->inds[mode].data;
    3f0c:	f940087b 	ldr	x27, [x3, #16]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    3f10:	fd0033e8 	str	d8, [sp, #96]
    3f14:	97fff4e7 	bl	12b0 <memset@plt>
	sptNewTimer(&timer, 0);
    3f18:	52800001 	mov	w1, #0x0                   	// #0
    3f1c:	910203e0 	add	x0, sp, #0x80
    3f20:	94000038 	bl	4000 <sptNewTimer>
	sptStartTimer(timer);
    3f24:	f94043e0 	ldr	x0, [sp, #128]
    3f28:	94000056 	bl	4080 <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    3f2c:	a908cff7 	stp	x23, x19, [sp, #136]
    3f30:	2a1503e2 	mov	w2, w21
    3f34:	f9403fe4 	ldr	x4, [sp, #120]
    3f38:	52800003 	mov	w3, #0x0                   	// #0
    3f3c:	910223e1 	add	x1, sp, #0x88
    3f40:	90000000 	adrp	x0, 3000 <sptRandomizeMatrix+0xc0>
    3f44:	91200000 	add	x0, x0, #0x800
    3f48:	a90993f6 	stp	x22, x4, [sp, #152]
    3f4c:	a90aeffc 	stp	x28, x27, [sp, #168]
    3f50:	f9005ff9 	str	x25, [sp, #184]
    3f54:	291863f4 	stp	w20, w24, [sp, #192]
    3f58:	b900cbfa 	str	w26, [sp, #200]
    3f5c:	97fff531 	bl	1420 <GOMP_parallel@plt>
	sptStopTimer(timer);
    3f60:	f94043e0 	ldr	x0, [sp, #128]
    3f64:	94000077 	bl	4140 <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    3f68:	f94043e0 	ldr	x0, [sp, #128]
    3f6c:	d0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    3f70:	91026021 	add	x1, x1, #0x98
    3f74:	940000b3 	bl	4240 <sptPrintElapsedTime>
    3f78:	1e604008 	fmov	d8, d0
	sptFreeTimer(timer);
    3f7c:	f94043e0 	ldr	x0, [sp, #128]
    3f80:	940000ec 	bl	4330 <sptFreeTimer>
	printf("[Total time]: %lf\n", total_time);
    3f84:	1e604100 	fmov	d0, d8
    3f88:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3f8c:	91016000 	add	x0, x0, #0x58
    3f90:	97fff51c 	bl	1400 <printf@plt>
	printf("\n");
    3f94:	52800140 	mov	w0, #0xa                   	// #10
    3f98:	97fff526 	bl	1430 <putchar@plt>
	return 0;
    3f9c:	52800000 	mov	w0, #0x0                   	// #0
    3fa0:	fd4033e8 	ldr	d8, [sp, #96]
}
    3fa4:	a94153f3 	ldp	x19, x20, [sp, #16]
	return 0;
    3fa8:	a9425bf5 	ldp	x21, x22, [sp, #32]
    3fac:	a94363f7 	ldp	x23, x24, [sp, #48]
    3fb0:	a9446bf9 	ldp	x25, x26, [sp, #64]
    3fb4:	a94573fb 	ldp	x27, x28, [sp, #80]
}
    3fb8:	a8cd7bfd 	ldp	x29, x30, [sp], #208
    3fbc:	d65f03c0 	ret
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    3fc0:	d0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    3fc4:	d0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    3fc8:	9100e084 	add	x4, x4, #0x38
    3fcc:	9101c042 	add	x2, x2, #0x70
    3fd0:	d0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    3fd4:	52800883 	mov	w3, #0x44                  	// #68
    3fd8:	17ffffb5 	b	3eac <sptOmpMTTKRP+0x98>
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    3fdc:	97ffff11 	bl	3c20 <sptOmpMTTKRP_3D>
    3fe0:	34fff720 	cbz	w0, 3ec4 <sptOmpMTTKRP+0xb0>
    3fe4:	12800000 	mov	w0, #0xffffffff            	// #-1
    3fe8:	a9025bf5 	stp	x21, x22, [sp, #32]
    3fec:	a90363f7 	stp	x23, x24, [sp, #48]
    3ff0:	a9046bf9 	stp	x25, x26, [sp, #64]
    3ff4:	a90573fb 	stp	x27, x28, [sp, #80]
    3ff8:	fd0033e8 	str	d8, [sp, #96]
    3ffc:	97fff471 	bl	11c0 <exit@plt>

0000000000004000 <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    4000:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4004:	910003fd 	mov	x29, sp
    4008:	a90153f3 	stp	x19, x20, [sp, #16]
    400c:	aa0003f4 	mov	x20, x0
    4010:	2a0103f3 	mov	w19, w1
	*timer = (sptTimer) malloc(sizeof **timer);
    4014:	d2800500 	mov	x0, #0x28                  	// #40
    4018:	97fff492 	bl	1260 <malloc@plt>
	(*timer)->use_cuda = use_cuda;
    401c:	b9000013 	str	w19, [x0]
	*timer = (sptTimer) malloc(sizeof **timer);
    4020:	f9000280 	str	x0, [x20]
	if(use_cuda) {
    4024:	350000b3 	cbnz	w19, 4038 <sptNewTimer+0x38>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	return 0;
}
    4028:	2a1303e0 	mov	w0, w19
    402c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4030:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4034:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4038:	52800073 	mov	w19, #0x3                   	// #3
    403c:	52800061 	mov	w1, #0x3                   	// #3
    4040:	72a00053 	movk	w19, #0x2, lsl #16
    4044:	72a00041 	movk	w1, #0x2, lsl #16
    4048:	528004e3 	mov	w3, #0x27                  	// #39
    404c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    4050:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    4054:	9102c084 	add	x4, x4, #0xb0
    4058:	91036042 	add	x2, x2, #0xd8
    405c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    4060:	91040000 	add	x0, x0, #0x100
    4064:	94000133 	bl	4530 <spt_ComplainError>
}
    4068:	2a1303e0 	mov	w0, w19
    406c:	a94153f3 	ldp	x19, x20, [sp, #16]
    4070:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4074:	d65f03c0 	ret
    4078:	d503201f 	nop
    407c:	d503201f 	nop

0000000000004080 <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    4080:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4084:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    4088:	b9400001 	ldr	w1, [x0]
    408c:	350000e1 	cbnz	w1, 40a8 <sptStartTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    4090:	91002001 	add	x1, x0, #0x8
    4094:	52800020 	mov	w0, #0x1                   	// #1
    4098:	97fff456 	bl	11f0 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    409c:	35000220 	cbnz	w0, 40e0 <sptStartTimer+0x60>
	}
	return 0;
}
    40a0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    40a4:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    40a8:	52800061 	mov	w1, #0x3                   	// #3
    40ac:	528005e3 	mov	w3, #0x2f                  	// #47
    40b0:	72a00041 	movk	w1, #0x2, lsl #16
    40b4:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    40b8:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    40bc:	9102c084 	add	x4, x4, #0xb0
    40c0:	91036042 	add	x2, x2, #0xd8
    40c4:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    40c8:	91040000 	add	x0, x0, #0x100
    40cc:	94000119 	bl	4530 <spt_ComplainError>
    40d0:	52800060 	mov	w0, #0x3                   	// #3
    40d4:	72a00040 	movk	w0, #0x2, lsl #16
}
    40d8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    40dc:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    40e0:	a90153f3 	stp	x19, x20, [sp, #16]
    40e4:	97fff4cb 	bl	1410 <__errno_location@plt>
    40e8:	aa0003f3 	mov	x19, x0
	return 0;
    40ec:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    40f0:	b9400274 	ldr	w20, [x19]
    40f4:	3140429f 	cmn	w20, #0x10, lsl #12
    40f8:	54000061 	b.ne	4104 <sptStartTimer+0x84>  // b.any
    40fc:	a94153f3 	ldp	x19, x20, [sp, #16]
    4100:	17ffffe8 	b	40a0 <sptStartTimer+0x20>
    4104:	2a1403e0 	mov	w0, w20
    4108:	97fff482 	bl	1310 <strerror@plt>
    410c:	11404281 	add	w1, w20, #0x10, lsl #12
    4110:	aa0003e4 	mov	x4, x0
    4114:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    4118:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    411c:	91036042 	add	x2, x2, #0xd8
    4120:	91040000 	add	x0, x0, #0x100
    4124:	52800643 	mov	w3, #0x32                  	// #50
    4128:	94000102 	bl	4530 <spt_ComplainError>
    412c:	b9400260 	ldr	w0, [x19]
    4130:	a94153f3 	ldp	x19, x20, [sp, #16]
    4134:	11404000 	add	w0, w0, #0x10, lsl #12
    4138:	17ffffda 	b	40a0 <sptStartTimer+0x20>
    413c:	d503201f 	nop

0000000000004140 <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    4140:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4144:	910003fd 	mov	x29, sp
	int result;
	if(timer->use_cuda) {
    4148:	b9400001 	ldr	w1, [x0]
    414c:	350000e1 	cbnz	w1, 4168 <sptStopTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    4150:	91006001 	add	x1, x0, #0x18
    4154:	52800020 	mov	w0, #0x1                   	// #1
    4158:	97fff426 	bl	11f0 <clock_gettime@plt>
		spt_CheckOSError(result, "Timer New");
    415c:	35000220 	cbnz	w0, 41a0 <sptStopTimer+0x60>
	}
	return 0;
}
    4160:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4164:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4168:	52800061 	mov	w1, #0x3                   	// #3
    416c:	52800743 	mov	w3, #0x3a                  	// #58
    4170:	72a00041 	movk	w1, #0x2, lsl #16
    4174:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    4178:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    417c:	9102c084 	add	x4, x4, #0xb0
    4180:	91036042 	add	x2, x2, #0xd8
    4184:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    4188:	91040000 	add	x0, x0, #0x100
    418c:	940000e9 	bl	4530 <spt_ComplainError>
    4190:	52800060 	mov	w0, #0x3                   	// #3
    4194:	72a00040 	movk	w0, #0x2, lsl #16
}
    4198:	a8c27bfd 	ldp	x29, x30, [sp], #32
    419c:	d65f03c0 	ret
		spt_CheckOSError(result, "Timer New");
    41a0:	a90153f3 	stp	x19, x20, [sp, #16]
    41a4:	97fff49b 	bl	1410 <__errno_location@plt>
    41a8:	aa0003f3 	mov	x19, x0
	return 0;
    41ac:	52800000 	mov	w0, #0x0                   	// #0
		spt_CheckOSError(result, "Timer New");
    41b0:	b9400274 	ldr	w20, [x19]
    41b4:	3140429f 	cmn	w20, #0x10, lsl #12
    41b8:	54000061 	b.ne	41c4 <sptStopTimer+0x84>  // b.any
    41bc:	a94153f3 	ldp	x19, x20, [sp, #16]
    41c0:	17ffffe8 	b	4160 <sptStopTimer+0x20>
    41c4:	2a1403e0 	mov	w0, w20
    41c8:	97fff452 	bl	1310 <strerror@plt>
    41cc:	11404281 	add	w1, w20, #0x10, lsl #12
    41d0:	aa0003e4 	mov	x4, x0
    41d4:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    41d8:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    41dc:	91036042 	add	x2, x2, #0xd8
    41e0:	91040000 	add	x0, x0, #0x100
    41e4:	528007a3 	mov	w3, #0x3d                  	// #61
    41e8:	940000d2 	bl	4530 <spt_ComplainError>
    41ec:	b9400260 	ldr	w0, [x19]
    41f0:	a94153f3 	ldp	x19, x20, [sp, #16]
    41f4:	11404000 	add	w0, w0, #0x10, lsl #12
    41f8:	17ffffda 	b	4160 <sptStopTimer+0x20>
    41fc:	d503201f 	nop

0000000000004200 <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
	if(timer->use_cuda) {
    4200:	b9400001 	ldr	w1, [x0]
    4204:	35000161 	cbnz	w1, 4230 <sptElapsedTime+0x30>
		return NAN;
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4208:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    420c:	a9408c04 	ldp	x4, x3, [x0, #8]
    4210:	fd409022 	ldr	d2, [x1, #288]
    4214:	a9418801 	ldp	x1, x2, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    4218:	cb040020 	sub	x0, x1, x4
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    421c:	cb030041 	sub	x1, x2, x3
    4220:	9e620021 	scvtf	d1, x1
    4224:	9e620000 	scvtf	d0, x0
    4228:	1f420020 	fmadd	d0, d1, d2, d0
	}
}
    422c:	d65f03c0 	ret
		return NAN;
    4230:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4234:	9e670000 	fmov	d0, x0
}
    4238:	d65f03c0 	ret
    423c:	d503201f 	nop

0000000000004240 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4240:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4244:	aa0103e2 	mov	x2, x1
    4248:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    424c:	b9400001 	ldr	w1, [x0]
double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    4250:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    4254:	350002a1 	cbnz	w1, 42a8 <sptPrintElapsedTime+0x68>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4258:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    425c:	a9409005 	ldp	x5, x4, [x0, #8]
    4260:	fd409021 	ldr	d1, [x1, #288]
    4264:	a9418403 	ldp	x3, x1, [x0, #24]
    4268:	cb040021 	sub	x1, x1, x4
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    426c:	cb050060 	sub	x0, x3, x5
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    4270:	9e620020 	scvtf	d0, x1
    4274:	9e620008 	scvtf	d8, x0
    4278:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer);
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    427c:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    4280:	1e604100 	fmov	d0, d8
    4284:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    4288:	91044021 	add	x1, x1, #0x110
    428c:	f947e000 	ldr	x0, [x0, #4032]
    4290:	f9400000 	ldr	x0, [x0]
    4294:	97fff46b 	bl	1440 <fprintf@plt>
	return elapsed_time;
}
    4298:	1e604100 	fmov	d0, d8
    429c:	fd400be8 	ldr	d8, [sp, #16]
    42a0:	a8c27bfd 	ldp	x29, x30, [sp], #32
    42a4:	d65f03c0 	ret
		return NAN;
    42a8:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    42ac:	9e670008 	fmov	d8, x0
    42b0:	17fffff3 	b	427c <sptPrintElapsedTime+0x3c>

00000000000042b4 <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    42b4:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    42b8:	910003fd 	mov	x29, sp
	if(timer->use_cuda) {
    42bc:	b9400003 	ldr	w3, [x0]
double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    42c0:	fd000be8 	str	d8, [sp, #16]
	if(timer->use_cuda) {
    42c4:	350002e3 	cbnz	w3, 4320 <sptPrintAverageElapsedTime+0x6c>
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    42c8:	b0000003 	adrp	x3, 5000 <_IO_stdin_used+0x7d8>
    42cc:	a9409406 	ldp	x6, x5, [x0, #8]
    42d0:	fd409061 	ldr	d1, [x3, #288]
    42d4:	a9419003 	ldp	x3, x4, [x0, #24]
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    42d8:	cb060060 	sub	x0, x3, x6
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    42dc:	cb050083 	sub	x3, x4, x5
    42e0:	9e620060 	scvtf	d0, x3
    42e4:	9e620008 	scvtf	d8, x0
    42e8:	1f412008 	fmadd	d8, d0, d1, d8
	double elapsed_time = sptElapsedTime(timer) / niters;
    42ec:	1e620020 	scvtf	d0, w1
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    42f0:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    42f4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    42f8:	91044021 	add	x1, x1, #0x110
    42fc:	f947e000 	ldr	x0, [x0, #4032]
	double elapsed_time = sptElapsedTime(timer) / niters;
    4300:	1e601908 	fdiv	d8, d8, d0
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    4304:	f9400000 	ldr	x0, [x0]
    4308:	1e604100 	fmov	d0, d8
    430c:	97fff44d 	bl	1440 <fprintf@plt>
	return elapsed_time;
}
    4310:	1e604100 	fmov	d0, d8
    4314:	fd400be8 	ldr	d8, [sp, #16]
    4318:	a8c27bfd 	ldp	x29, x30, [sp], #32
    431c:	d65f03c0 	ret
		return NAN;
    4320:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    4324:	9e670008 	fmov	d8, x0
    4328:	17fffff1 	b	42ec <sptPrintAverageElapsedTime+0x38>
    432c:	d503201f 	nop

0000000000004330 <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    4330:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4334:	910003fd 	mov	x29, sp
    4338:	f9000bf3 	str	x19, [sp, #16]
	if(timer->use_cuda) {
    433c:	b9400013 	ldr	w19, [x0]
    4340:	350000d3 	cbnz	w19, 4358 <sptFreeTimer+0x28>
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
	}
	free(timer);
    4344:	97fff417 	bl	13a0 <free@plt>
	return 0;
}
    4348:	2a1303e0 	mov	w0, w19
    434c:	f9400bf3 	ldr	x19, [sp, #16]
    4350:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4354:	d65f03c0 	ret
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    4358:	52800061 	mov	w1, #0x3                   	// #3
    435c:	52800073 	mov	w19, #0x3                   	// #3
    4360:	72a00041 	movk	w1, #0x2, lsl #16
    4364:	52800b63 	mov	w3, #0x5b                  	// #91
    4368:	72a00053 	movk	w19, #0x2, lsl #16
    436c:	b0000004 	adrp	x4, 5000 <_IO_stdin_used+0x7d8>
    4370:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    4374:	9102c084 	add	x4, x4, #0xb0
    4378:	91036042 	add	x2, x2, #0xd8
    437c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    4380:	91040000 	add	x0, x0, #0x100
    4384:	9400006b 	bl	4530 <spt_ComplainError>
}
    4388:	2a1303e0 	mov	w0, w19
    438c:	f9400bf3 	ldr	x19, [sp, #16]
    4390:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4394:	d65f03c0 	ret
    4398:	d503201f 	nop
    439c:	d503201f 	nop

00000000000043a0 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    43a0:	a9b97bfd 	stp	x29, x30, [sp, #-112]!
    43a4:	910003fd 	mov	x29, sp
    43a8:	a9025bf5 	stp	x21, x22, [sp, #32]
    43ac:	aa0003f6 	mov	x22, x0
    43b0:	aa0103f5 	mov	x21, x1
    43b4:	a9046bf9 	stp	x25, x26, [sp, #64]
	int iores;
	sptIndex nrows = mtx->nrows;
	sptIndex ncols = mtx->ncols;
	sptIndex stride = mtx->stride;
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    43b8:	aa0103e0 	mov	x0, x1
    43bc:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
	sptIndex ncols = mtx->ncols;
    43c0:	294066da 	ldp	w26, w25, [x22]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    43c4:	9104a021 	add	x1, x1, #0x128
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    43c8:	a90153f3 	stp	x19, x20, [sp, #16]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    43cc:	2a1a03e2 	mov	w2, w26
    43d0:	2a1903e3 	mov	w3, w25
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    43d4:	a90573fb 	stp	x27, x28, [sp, #80]
	sptIndex stride = mtx->stride;
    43d8:	b9400edb 	ldr	w27, [x22, #12]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    43dc:	97fff419 	bl	1440 <fprintf@plt>
	spt_CheckOSError(iores < 0, "Mtx Dump");
    43e0:	37f80820 	tbnz	w0, #31, 44e4 <sptDumpMatrix+0x144>
	for(sptIndex i=0; i < nrows; ++i) {
    43e4:	a90363f7 	stp	x23, x24, [sp, #48]
	sptIndex ncols = mtx->ncols;
    43e8:	2a1903f4 	mov	w20, w25
    43ec:	b0000017 	adrp	x23, 5000 <_IO_stdin_used+0x7d8>
	for(sptIndex i=0; i < nrows; ++i) {
    43f0:	52800018 	mov	w24, #0x0                   	// #0
		for(sptIndex j=0; j < ncols; ++j) {
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    43f4:	9105c2f7 	add	x23, x23, #0x170
	for(sptIndex i=0; i < nrows; ++i) {
    43f8:	3400061a 	cbz	w26, 44b8 <sptDumpMatrix+0x118>
    43fc:	d503201f 	nop
		for(sptIndex j=0; j < ncols; ++j) {
    4400:	4b190293 	sub	w19, w20, w25
    4404:	35000099 	cbnz	w25, 4414 <sptDumpMatrix+0x74>
    4408:	14000025 	b	449c <sptDumpMatrix+0xfc>
    440c:	6b13029f 	cmp	w20, w19
    4410:	54000460 	b.eq	449c <sptDumpMatrix+0xfc>  // b.none
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4414:	f9400ac2 	ldr	x2, [x22, #16]
    4418:	aa1703e1 	mov	x1, x23
    441c:	aa1503e0 	mov	x0, x21
    4420:	bc735840 	ldr	s0, [x2, w19, uxtw #2]
		for(sptIndex j=0; j < ncols; ++j) {
    4424:	11000673 	add	w19, w19, #0x1
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    4428:	1e22c000 	fcvt	d0, s0
    442c:	97fff405 	bl	1440 <fprintf@plt>
			spt_CheckOSError(iores < 0, "Mtx Dump");
    4430:	36fffee0 	tbz	w0, #31, 440c <sptDumpMatrix+0x6c>
    4434:	97fff3f7 	bl	1410 <__errno_location@plt>
    4438:	b9400001 	ldr	w1, [x0]
    443c:	aa0003fc 	mov	x28, x0
    4440:	3140403f 	cmn	w1, #0x10, lsl #12
    4444:	54fffe40 	b.eq	440c <sptDumpMatrix+0x6c>  // b.none
    4448:	2a0103e0 	mov	w0, w1
    444c:	b9006fe1 	str	w1, [sp, #108]
    4450:	97fff3b0 	bl	1310 <strerror@plt>
    4454:	aa0003e4 	mov	x4, x0
    4458:	b9406fe1 	ldr	w1, [sp, #108]
    445c:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    4460:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    4464:	9104e042 	add	x2, x2, #0x138
    4468:	91058000 	add	x0, x0, #0x160
    446c:	11404021 	add	w1, w1, #0x10, lsl #12
    4470:	528005c3 	mov	w3, #0x2e                  	// #46
    4474:	9400002f 	bl	4530 <spt_ComplainError>
    4478:	a94363f7 	ldp	x23, x24, [sp, #48]
    447c:	b9400380 	ldr	w0, [x28]
    4480:	11404000 	add	w0, w0, #0x10, lsl #12
		}
		iores = fprintf(fp, "\n");
	}
	iores = fprintf(fp, "\n");
	return 0;
}
    4484:	a94153f3 	ldp	x19, x20, [sp, #16]
    4488:	a9425bf5 	ldp	x21, x22, [sp, #32]
    448c:	a9446bf9 	ldp	x25, x26, [sp, #64]
    4490:	a94573fb 	ldp	x27, x28, [sp, #80]
    4494:	a8c77bfd 	ldp	x29, x30, [sp], #112
    4498:	d65f03c0 	ret
		iores = fprintf(fp, "\n");
    449c:	aa1503e1 	mov	x1, x21
    44a0:	52800140 	mov	w0, #0xa                   	// #10
	for(sptIndex i=0; i < nrows; ++i) {
    44a4:	11000718 	add	w24, w24, #0x1
		iores = fprintf(fp, "\n");
    44a8:	97fff34e 	bl	11e0 <fputc@plt>
	for(sptIndex i=0; i < nrows; ++i) {
    44ac:	0b1b0294 	add	w20, w20, w27
    44b0:	6b18035f 	cmp	w26, w24
    44b4:	54fffa61 	b.ne	4400 <sptDumpMatrix+0x60>  // b.any
	iores = fprintf(fp, "\n");
    44b8:	aa1503e1 	mov	x1, x21
    44bc:	52800140 	mov	w0, #0xa                   	// #10
    44c0:	97fff348 	bl	11e0 <fputc@plt>
	return 0;
    44c4:	52800000 	mov	w0, #0x0                   	// #0
}
    44c8:	a94153f3 	ldp	x19, x20, [sp, #16]
    44cc:	a9425bf5 	ldp	x21, x22, [sp, #32]
	return 0;
    44d0:	a94363f7 	ldp	x23, x24, [sp, #48]
}
    44d4:	a9446bf9 	ldp	x25, x26, [sp, #64]
    44d8:	a94573fb 	ldp	x27, x28, [sp, #80]
    44dc:	a8c77bfd 	ldp	x29, x30, [sp], #112
    44e0:	d65f03c0 	ret
	spt_CheckOSError(iores < 0, "Mtx Dump");
    44e4:	97fff3cb 	bl	1410 <__errno_location@plt>
    44e8:	b9400014 	ldr	w20, [x0]
    44ec:	aa0003f3 	mov	x19, x0
    44f0:	3140429f 	cmn	w20, #0x10, lsl #12
    44f4:	54fff780 	b.eq	43e4 <sptDumpMatrix+0x44>  // b.none
    44f8:	2a1403e0 	mov	w0, w20
    44fc:	97fff385 	bl	1310 <strerror@plt>
    4500:	11404281 	add	w1, w20, #0x10, lsl #12
    4504:	aa0003e4 	mov	x4, x0
    4508:	b0000002 	adrp	x2, 5000 <_IO_stdin_used+0x7d8>
    450c:	b0000000 	adrp	x0, 5000 <_IO_stdin_used+0x7d8>
    4510:	9104e042 	add	x2, x2, #0x138
    4514:	91058000 	add	x0, x0, #0x160
    4518:	52800543 	mov	w3, #0x2a                  	// #42
    451c:	94000005 	bl	4530 <spt_ComplainError>
    4520:	b9400260 	ldr	w0, [x19]
    4524:	11404000 	add	w0, w0, #0x10, lsl #12
    4528:	17ffffd7 	b	4484 <sptDumpMatrix+0xe4>
    452c:	d503201f 	nop

0000000000004530 <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4530:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4534:	910003fd 	mov	x29, sp
    4538:	a90153f3 	stp	x19, x20, [sp, #16]
	g_last_error.errcode = errcode;
    453c:	d53bd053 	mrs	x19, tpidr_el0
    4540:	91400273 	add	x19, x19, #0x0, lsl #12
    4544:	91004273 	add	x19, x19, #0x10
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    4548:	a9025bf5 	stp	x21, x22, [sp, #32]
    454c:	aa0003f4 	mov	x20, x0
    4550:	2a0103f5 	mov	w21, w1
    4554:	a90363f7 	stp	x23, x24, [sp, #48]
    4558:	aa0203f6 	mov	x22, x2
    455c:	2a0303f7 	mov	w23, w3
	g_last_error.module = module;
    4560:	f9000260 	str	x0, [x19]
	g_last_error.errcode = errcode;
    4564:	b9000a61 	str	w1, [x19, #8]
	g_last_error.file = file;
    4568:	f9000a62 	str	x2, [x19, #16]
	g_last_error.line = line;
    456c:	b9001a63 	str	w3, [x19, #24]
	if(reason) {
		free(g_last_error.reason);
    4570:	f9401266 	ldr	x6, [x19, #32]
	if(reason) {
    4574:	b4000324 	cbz	x4, 45d8 <spt_ComplainError+0xa8>
		free(g_last_error.reason);
    4578:	aa0603e0 	mov	x0, x6
    457c:	aa0403f8 	mov	x24, x4
    4580:	97fff388 	bl	13a0 <free@plt>
		g_last_error.reason = strdup(reason);
    4584:	aa1803e0 	mov	x0, x24
    4588:	97fff35a 	bl	12f0 <strdup@plt>
    458c:	f9001260 	str	x0, [x19, #32]
    4590:	aa0003e6 	mov	x6, x0
		if(!g_last_error.reason) {
    4594:	b4000440 	cbz	x0, 461c <spt_ComplainError+0xec>
			abort();
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4598:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x10088>
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    459c:	394000c1 	ldrb	w1, [x6]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    45a0:	f947d800 	ldr	x0, [x0, #4016]
    45a4:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    45a8:	35000241 	cbnz	w1, 45f0 <spt_ComplainError+0xc0>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    45ac:	2a1703e5 	mov	w5, w23
    45b0:	aa1603e4 	mov	x4, x22
    45b4:	2a1503e3 	mov	w3, w21
    45b8:	aa1403e2 	mov	x2, x20
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    45bc:	a94153f3 	ldp	x19, x20, [sp, #16]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    45c0:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
}
    45c4:	a9425bf5 	ldp	x21, x22, [sp, #32]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    45c8:	91066021 	add	x1, x1, #0x198
}
    45cc:	a94363f7 	ldp	x23, x24, [sp, #48]
    45d0:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    45d4:	17fff39b 	b	1440 <fprintf@plt>
    45d8:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    45dc:	f947d800 	ldr	x0, [x0, #4016]
    45e0:	f9400000 	ldr	x0, [x0]
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    45e4:	b4fffe46 	cbz	x6, 45ac <spt_ComplainError+0x7c>
    45e8:	394000c1 	ldrb	w1, [x6]
    45ec:	34fffe01 	cbz	w1, 45ac <spt_ComplainError+0x7c>
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    45f0:	2a1703e5 	mov	w5, w23
    45f4:	aa1603e4 	mov	x4, x22
    45f8:	2a1503e3 	mov	w3, w21
    45fc:	aa1403e2 	mov	x2, x20
}
    4600:	a94153f3 	ldp	x19, x20, [sp, #16]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4604:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
}
    4608:	a9425bf5 	ldp	x21, x22, [sp, #32]
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    460c:	9105e021 	add	x1, x1, #0x178
}
    4610:	a94363f7 	ldp	x23, x24, [sp, #48]
    4614:	a8c47bfd 	ldp	x29, x30, [sp], #64
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    4618:	17fff38a 	b	1440 <fprintf@plt>
			abort();
    461c:	97fff34d 	bl	1350 <abort@plt>

0000000000004620 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    4620:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
	double size = (double)bytes;
	int suff = 0;
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
	while(size > 1024 && suff < 5) {
    4624:	d2e81201 	mov	x1, #0x4090000000000000    	// #4652218415073722368
    4628:	9e670021 	fmov	d1, x1
{
    462c:	910003fd 	mov	x29, sp
    4630:	fd000be8 	str	d8, [sp, #16]
    4634:	9e670008 	fmov	d8, x0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4638:	f0000081 	adrp	x1, 17000 <memcpy@GLIBC_2.17>
    463c:	910a0021 	add	x1, x1, #0x280
	double size = (double)bytes;
    4640:	7e61d900 	ucvtf	d0, d8
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4644:	9100a3e2 	add	x2, sp, #0x28
		size /= 1024.;
    4648:	d2e7ea00 	mov	x0, #0x3f50000000000000    	// #4562146422526312448
    464c:	9e670002 	fmov	d2, x0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4650:	a9401424 	ldp	x4, x5, [x1]
    4654:	a90297e4 	stp	x4, x5, [sp, #40]
	while(size > 1024 && suff < 5) {
    4658:	1e612010 	fcmpe	d0, d1
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    465c:	a9411424 	ldp	x4, x5, [x1, #16]
    4660:	a9011444 	stp	x4, x5, [x2, #16]
	int suff = 0;
    4664:	52800000 	mov	w0, #0x0                   	// #0
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    4668:	f9401021 	ldr	x1, [x1, #32]
    466c:	f9001041 	str	x1, [x2, #32]
	while(size > 1024 && suff < 5) {
    4670:	5400020c 	b.gt	46b0 <sptBytesString+0x90>
    4674:	d503201f 	nop
		++suff;
	}
	char * ret = NULL;
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    4678:	f860d842 	ldr	x2, [x2, w0, sxtw #3]
    467c:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    4680:	910083e0 	add	x0, sp, #0x20
    4684:	91078021 	add	x1, x1, #0x1e0
	char * ret = NULL;
    4688:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    468c:	97fff2dd 	bl	1200 <asprintf@plt>
    4690:	3100041f 	cmn	w0, #0x1
    4694:	54000260 	b.eq	46e0 <sptBytesString+0xc0>  // b.none
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
		ret = NULL;
	}
	return ret;
}
    4698:	fd400be8 	ldr	d8, [sp, #16]
	return ret;
    469c:	f94013e0 	ldr	x0, [sp, #32]
}
    46a0:	a8c57bfd 	ldp	x29, x30, [sp], #80
    46a4:	d65f03c0 	ret
	while(size > 1024 && suff < 5) {
    46a8:	7100141f 	cmp	w0, #0x5
    46ac:	54fffe60 	b.eq	4678 <sptBytesString+0x58>  // b.none
		size /= 1024.;
    46b0:	1e620800 	fmul	d0, d0, d2
		++suff;
    46b4:	11000400 	add	w0, w0, #0x1
	while(size > 1024 && suff < 5) {
    46b8:	1e612010 	fcmpe	d0, d1
    46bc:	54ffff6c 	b.gt	46a8 <sptBytesString+0x88>
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    46c0:	f860d842 	ldr	x2, [x2, w0, sxtw #3]
    46c4:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    46c8:	910083e0 	add	x0, sp, #0x20
    46cc:	91078021 	add	x1, x1, #0x1e0
	char * ret = NULL;
    46d0:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    46d4:	97fff2cb 	bl	1200 <asprintf@plt>
    46d8:	3100041f 	cmn	w0, #0x1
    46dc:	54fffde1 	b.ne	4698 <sptBytesString+0x78>  // b.any
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    46e0:	d0000080 	adrp	x0, 16000 <__FRAME_END__+0x10088>
    46e4:	9e660102 	fmov	x2, d8
    46e8:	b0000001 	adrp	x1, 5000 <_IO_stdin_used+0x7d8>
    46ec:	9107a021 	add	x1, x1, #0x1e8
    46f0:	f947d800 	ldr	x0, [x0, #4016]
    46f4:	f9400000 	ldr	x0, [x0]
    46f8:	97fff352 	bl	1440 <fprintf@plt>
    46fc:	d2800000 	mov	x0, #0x0                   	// #0
}
    4700:	fd400be8 	ldr	d8, [sp, #16]
    4704:	a8c57bfd 	ldp	x29, x30, [sp], #80
    4708:	d65f03c0 	ret
    470c:	d503201f 	nop

0000000000004710 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    4710:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    4714:	910003fd 	mov	x29, sp
    4718:	fd000be8 	str	d8, [sp, #16]
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    471c:	97fff2f1 	bl	12e0 <rand@plt>
    4720:	1e028408 	scvtf	s8, w0, #31
    4724:	1e211000 	fmov	s0, #3.000000000000000000e+00
    4728:	1e200908 	fmul	s8, s8, s0
	if(rand() % 2 == 0) {
    472c:	97fff2ed 	bl	12e0 <rand@plt>
		v *= -1;
    4730:	1e214100 	fneg	s0, s8
    4734:	f240001f 	tst	x0, #0x1
	}
	return v;
}
    4738:	1e280c00 	fcsel	s0, s0, s8, eq  // eq = none
    473c:	fd400be8 	ldr	d8, [sp, #16]
    4740:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4744:	d65f03c0 	ret
    4748:	d503201f 	nop
    474c:	d503201f 	nop

0000000000004750 <__aarch64_cas4_acq_rel>:
    4750:	d503245f 	bti	c
    4754:	f0000090 	adrp	x16, 17000 <memcpy@GLIBC_2.17>
    4758:	394aa610 	ldrb	w16, [x16, #681]
    475c:	34000070 	cbz	w16, 4768 <__aarch64_cas4_acq_rel+0x18>
    4760:	88e0fc41 	casal	w0, w1, [x2]
    4764:	d65f03c0 	ret
    4768:	2a0003f0 	mov	w16, w0
    476c:	885ffc40 	ldaxr	w0, [x2]
    4770:	6b10001f 	cmp	w0, w16
    4774:	54000061 	b.ne	4780 <__aarch64_cas4_acq_rel+0x30>  // b.any
    4778:	8811fc41 	stlxr	w17, w1, [x2]
    477c:	35ffff91 	cbnz	w17, 476c <__aarch64_cas4_acq_rel+0x1c>
    4780:	d65f03c0 	ret
    4784:	d503201f 	nop
    4788:	d503201f 	nop
    478c:	d503201f 	nop

0000000000004790 <__libc_csu_init>:
    4790:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    4794:	910003fd 	mov	x29, sp
    4798:	a90153f3 	stp	x19, x20, [sp, #16]
    479c:	d0000094 	adrp	x20, 16000 <__FRAME_END__+0x10088>
    47a0:	9134e294 	add	x20, x20, #0xd38
    47a4:	a9025bf5 	stp	x21, x22, [sp, #32]
    47a8:	d0000095 	adrp	x21, 16000 <__FRAME_END__+0x10088>
    47ac:	9134a2b5 	add	x21, x21, #0xd28
    47b0:	cb150294 	sub	x20, x20, x21
    47b4:	2a0003f6 	mov	w22, w0
    47b8:	a90363f7 	stp	x23, x24, [sp, #48]
    47bc:	aa0103f7 	mov	x23, x1
    47c0:	aa0203f8 	mov	x24, x2
    47c4:	9343fe94 	asr	x20, x20, #3
    47c8:	97fff26c 	bl	1178 <_init>
    47cc:	b4000174 	cbz	x20, 47f8 <__libc_csu_init+0x68>
    47d0:	d2800013 	mov	x19, #0x0                   	// #0
    47d4:	d503201f 	nop
    47d8:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    47dc:	aa1803e2 	mov	x2, x24
    47e0:	91000673 	add	x19, x19, #0x1
    47e4:	aa1703e1 	mov	x1, x23
    47e8:	2a1603e0 	mov	w0, w22
    47ec:	d63f0060 	blr	x3
    47f0:	eb13029f 	cmp	x20, x19
    47f4:	54ffff21 	b.ne	47d8 <__libc_csu_init+0x48>  // b.any
    47f8:	a94153f3 	ldp	x19, x20, [sp, #16]
    47fc:	a9425bf5 	ldp	x21, x22, [sp, #32]
    4800:	a94363f7 	ldp	x23, x24, [sp, #48]
    4804:	a8c47bfd 	ldp	x29, x30, [sp], #64
    4808:	d65f03c0 	ret
    480c:	d503201f 	nop

0000000000004810 <__libc_csu_fini>:
    4810:	d65f03c0 	ret

Disassembly of section .fini:

0000000000004814 <_fini>:
    4814:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    4818:	910003fd 	mov	x29, sp
    481c:	a8c17bfd 	ldp	x29, x30, [sp], #16
    4820:	d65f03c0 	ret
