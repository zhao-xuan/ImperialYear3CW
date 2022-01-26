
mttkrp:     file format elf64-littleaarch64


Disassembly of section .init:

0000000000001138 <_init>:
    1138:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    113c:	910003fd 	mov	x29, sp
    1140:	940000c7 	bl	145c <call_weak_fn>
    1144:	a8c17bfd 	ldp	x29, x30, [sp], #16
    1148:	d65f03c0 	ret

Disassembly of section .plt:

0000000000001150 <.plt>:
    1150:	a9bf7bf0 	stp	x16, x30, [sp, #-16]!
    1154:	d00000b0 	adrp	x16, 17000 <__FRAME_END__+0xf8b8>
    1158:	f947fe11 	ldr	x17, [x16, #4088]
    115c:	913fe210 	add	x16, x16, #0xff8
    1160:	d61f0220 	br	x17
    1164:	d503201f 	nop
    1168:	d503201f 	nop
    116c:	d503201f 	nop

0000000000001170 <memcpy@plt>:
    1170:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1174:	f9400211 	ldr	x17, [x16]
    1178:	91000210 	add	x16, x16, #0x0
    117c:	d61f0220 	br	x17

0000000000001180 <exit@plt>:
    1180:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1184:	f9400611 	ldr	x17, [x16, #8]
    1188:	91002210 	add	x16, x16, #0x8
    118c:	d61f0220 	br	x17

0000000000001190 <__cxa_finalize@plt>:
    1190:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1194:	f9400a11 	ldr	x17, [x16, #16]
    1198:	91004210 	add	x16, x16, #0x10
    119c:	d61f0220 	br	x17

00000000000011a0 <fputc@plt>:
    11a0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    11a4:	f9400e11 	ldr	x17, [x16, #24]
    11a8:	91006210 	add	x16, x16, #0x18
    11ac:	d61f0220 	br	x17

00000000000011b0 <clock_gettime@plt>:
    11b0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    11b4:	f9401211 	ldr	x17, [x16, #32]
    11b8:	91008210 	add	x16, x16, #0x20
    11bc:	d61f0220 	br	x17

00000000000011c0 <asprintf@plt>:
    11c0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    11c4:	f9401611 	ldr	x17, [x16, #40]
    11c8:	9100a210 	add	x16, x16, #0x28
    11cc:	d61f0220 	br	x17

00000000000011d0 <pow@plt>:
    11d0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    11d4:	f9401a11 	ldr	x17, [x16, #48]
    11d8:	9100c210 	add	x16, x16, #0x30
    11dc:	d61f0220 	br	x17

00000000000011e0 <omp_get_thread_num@plt>:
    11e0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    11e4:	f9401e11 	ldr	x17, [x16, #56]
    11e8:	9100e210 	add	x16, x16, #0x38
    11ec:	d61f0220 	br	x17

00000000000011f0 <fclose@plt>:
    11f0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    11f4:	f9402211 	ldr	x17, [x16, #64]
    11f8:	91010210 	add	x16, x16, #0x40
    11fc:	d61f0220 	br	x17

0000000000001200 <fopen@plt>:
    1200:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1204:	f9402611 	ldr	x17, [x16, #72]
    1208:	91012210 	add	x16, x16, #0x48
    120c:	d61f0220 	br	x17

0000000000001210 <time@plt>:
    1210:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1214:	f9402a11 	ldr	x17, [x16, #80]
    1218:	91014210 	add	x16, x16, #0x50
    121c:	d61f0220 	br	x17

0000000000001220 <malloc@plt>:
    1220:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1224:	f9402e11 	ldr	x17, [x16, #88]
    1228:	91016210 	add	x16, x16, #0x58
    122c:	d61f0220 	br	x17

0000000000001230 <__isoc99_fscanf@plt>:
    1230:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1234:	f9403211 	ldr	x17, [x16, #96]
    1238:	91018210 	add	x16, x16, #0x60
    123c:	d61f0220 	br	x17

0000000000001240 <__libc_start_main@plt>:
    1240:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1244:	f9403611 	ldr	x17, [x16, #104]
    1248:	9101a210 	add	x16, x16, #0x68
    124c:	d61f0220 	br	x17

0000000000001250 <fgetc@plt>:
    1250:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1254:	f9403a11 	ldr	x17, [x16, #112]
    1258:	9101c210 	add	x16, x16, #0x70
    125c:	d61f0220 	br	x17

0000000000001260 <aligned_alloc@plt>:
    1260:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1264:	f9403e11 	ldr	x17, [x16, #120]
    1268:	9101e210 	add	x16, x16, #0x78
    126c:	d61f0220 	br	x17

0000000000001270 <memset@plt>:
    1270:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1274:	f9404211 	ldr	x17, [x16, #128]
    1278:	91020210 	add	x16, x16, #0x80
    127c:	d61f0220 	br	x17

0000000000001280 <realloc@plt>:
    1280:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1284:	f9404611 	ldr	x17, [x16, #136]
    1288:	91022210 	add	x16, x16, #0x88
    128c:	d61f0220 	br	x17

0000000000001290 <rand@plt>:
    1290:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1294:	f9404a11 	ldr	x17, [x16, #144]
    1298:	91024210 	add	x16, x16, #0x90
    129c:	d61f0220 	br	x17

00000000000012a0 <strdup@plt>:
    12a0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    12a4:	f9404e11 	ldr	x17, [x16, #152]
    12a8:	91026210 	add	x16, x16, #0x98
    12ac:	d61f0220 	br	x17

00000000000012b0 <omp_get_num_threads@plt>:
    12b0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    12b4:	f9405211 	ldr	x17, [x16, #160]
    12b8:	91028210 	add	x16, x16, #0xa0
    12bc:	d61f0220 	br	x17

00000000000012c0 <strerror@plt>:
    12c0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    12c4:	f9405611 	ldr	x17, [x16, #168]
    12c8:	9102a210 	add	x16, x16, #0xa8
    12cc:	d61f0220 	br	x17

00000000000012d0 <strrchr@plt>:
    12d0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    12d4:	f9405a11 	ldr	x17, [x16, #176]
    12d8:	9102c210 	add	x16, x16, #0xb0
    12dc:	d61f0220 	br	x17

00000000000012e0 <__gmon_start__@plt>:
    12e0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    12e4:	f9405e11 	ldr	x17, [x16, #184]
    12e8:	9102e210 	add	x16, x16, #0xb8
    12ec:	d61f0220 	br	x17

00000000000012f0 <__getauxval@plt>:
    12f0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    12f4:	f9406211 	ldr	x17, [x16, #192]
    12f8:	91030210 	add	x16, x16, #0xc0
    12fc:	d61f0220 	br	x17

0000000000001300 <abort@plt>:
    1300:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1304:	f9406611 	ldr	x17, [x16, #200]
    1308:	91032210 	add	x16, x16, #0xc8
    130c:	d61f0220 	br	x17

0000000000001310 <puts@plt>:
    1310:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1314:	f9406a11 	ldr	x17, [x16, #208]
    1318:	91034210 	add	x16, x16, #0xd0
    131c:	d61f0220 	br	x17

0000000000001320 <getopt_long@plt>:
    1320:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1324:	f9406e11 	ldr	x17, [x16, #216]
    1328:	91036210 	add	x16, x16, #0xd8
    132c:	d61f0220 	br	x17

0000000000001330 <strcmp@plt>:
    1330:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1334:	f9407211 	ldr	x17, [x16, #224]
    1338:	91038210 	add	x16, x16, #0xe0
    133c:	d61f0220 	br	x17

0000000000001340 <fread@plt>:
    1340:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1344:	f9407611 	ldr	x17, [x16, #232]
    1348:	9103a210 	add	x16, x16, #0xe8
    134c:	d61f0220 	br	x17

0000000000001350 <free@plt>:
    1350:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1354:	f9407a11 	ldr	x17, [x16, #240]
    1358:	9103c210 	add	x16, x16, #0xf0
    135c:	d61f0220 	br	x17

0000000000001360 <fwrite@plt>:
    1360:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1364:	f9407e11 	ldr	x17, [x16, #248]
    1368:	9103e210 	add	x16, x16, #0xf8
    136c:	d61f0220 	br	x17

0000000000001370 <fflush@plt>:
    1370:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1374:	f9408211 	ldr	x17, [x16, #256]
    1378:	91040210 	add	x16, x16, #0x100
    137c:	d61f0220 	br	x17

0000000000001380 <strcpy@plt>:
    1380:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1384:	f9408611 	ldr	x17, [x16, #264]
    1388:	91042210 	add	x16, x16, #0x108
    138c:	d61f0220 	br	x17

0000000000001390 <srand@plt>:
    1390:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    1394:	f9408a11 	ldr	x17, [x16, #272]
    1398:	91044210 	add	x16, x16, #0x110
    139c:	d61f0220 	br	x17

00000000000013a0 <__isoc99_sscanf@plt>:
    13a0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    13a4:	f9408e11 	ldr	x17, [x16, #280]
    13a8:	91046210 	add	x16, x16, #0x118
    13ac:	d61f0220 	br	x17

00000000000013b0 <printf@plt>:
    13b0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    13b4:	f9409211 	ldr	x17, [x16, #288]
    13b8:	91048210 	add	x16, x16, #0x120
    13bc:	d61f0220 	br	x17

00000000000013c0 <__errno_location@plt>:
    13c0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    13c4:	f9409611 	ldr	x17, [x16, #296]
    13c8:	9104a210 	add	x16, x16, #0x128
    13cc:	d61f0220 	br	x17

00000000000013d0 <GOMP_parallel@plt>:
    13d0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    13d4:	f9409a11 	ldr	x17, [x16, #304]
    13d8:	9104c210 	add	x16, x16, #0x130
    13dc:	d61f0220 	br	x17

00000000000013e0 <putchar@plt>:
    13e0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    13e4:	f9409e11 	ldr	x17, [x16, #312]
    13e8:	9104e210 	add	x16, x16, #0x138
    13ec:	d61f0220 	br	x17

00000000000013f0 <fprintf@plt>:
    13f0:	f00000b0 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    13f4:	f940a211 	ldr	x17, [x16, #320]
    13f8:	91050210 	add	x16, x16, #0x140
    13fc:	d61f0220 	br	x17

Disassembly of section .text:

0000000000001400 <init_have_lse_atomics>:
    1400:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    1404:	d2800200 	mov	x0, #0x10                  	// #16
    1408:	910003fd 	mov	x29, sp
    140c:	97ffffb9 	bl	12f0 <__getauxval@plt>
    1410:	53082000 	ubfx	w0, w0, #8, #1
    1414:	f00000a1 	adrp	x1, 18000 <memcpy@GLIBC_2.17>
    1418:	a8c17bfd 	ldp	x29, x30, [sp], #16
    141c:	390b8420 	strb	w0, [x1, #737]
    1420:	d65f03c0 	ret

0000000000001424 <_start>:
    1424:	d280001d 	mov	x29, #0x0                   	// #0
    1428:	d280001e 	mov	x30, #0x0                   	// #0
    142c:	aa0003e5 	mov	x5, x0
    1430:	f94003e1 	ldr	x1, [sp]
    1434:	910023e2 	add	x2, sp, #0x8
    1438:	910003e6 	mov	x6, sp
    143c:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    1440:	f947ec00 	ldr	x0, [x0, #4056]
    1444:	d00000a3 	adrp	x3, 17000 <__FRAME_END__+0xf8b8>
    1448:	f947e863 	ldr	x3, [x3, #4048]
    144c:	d00000a4 	adrp	x4, 17000 <__FRAME_END__+0xf8b8>
    1450:	f947cc84 	ldr	x4, [x4, #3992]
    1454:	97ffff7b 	bl	1240 <__libc_start_main@plt>
    1458:	97ffffaa 	bl	1300 <abort@plt>

000000000000145c <call_weak_fn>:
    145c:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    1460:	f947e400 	ldr	x0, [x0, #4040]
    1464:	b4000040 	cbz	x0, 146c <call_weak_fn+0x10>
    1468:	17ffff9e 	b	12e0 <__gmon_start__@plt>
    146c:	d65f03c0 	ret

0000000000001470 <deregister_tm_clones>:
    1470:	f00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    1474:	910b8000 	add	x0, x0, #0x2e0
    1478:	f00000a1 	adrp	x1, 18000 <memcpy@GLIBC_2.17>
    147c:	910b8021 	add	x1, x1, #0x2e0
    1480:	eb00003f 	cmp	x1, x0
    1484:	540000c0 	b.eq	149c <deregister_tm_clones+0x2c>  // b.none
    1488:	d00000a1 	adrp	x1, 17000 <__FRAME_END__+0xf8b8>
    148c:	f947d021 	ldr	x1, [x1, #4000]
    1490:	b4000061 	cbz	x1, 149c <deregister_tm_clones+0x2c>
    1494:	aa0103f0 	mov	x16, x1
    1498:	d61f0200 	br	x16
    149c:	d65f03c0 	ret

00000000000014a0 <register_tm_clones>:
    14a0:	f00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    14a4:	910b8000 	add	x0, x0, #0x2e0
    14a8:	f00000a1 	adrp	x1, 18000 <memcpy@GLIBC_2.17>
    14ac:	910b8021 	add	x1, x1, #0x2e0
    14b0:	cb000021 	sub	x1, x1, x0
    14b4:	d37ffc22 	lsr	x2, x1, #63
    14b8:	8b810c41 	add	x1, x2, x1, asr #3
    14bc:	9341fc21 	asr	x1, x1, #1
    14c0:	b40000c1 	cbz	x1, 14d8 <register_tm_clones+0x38>
    14c4:	d00000a2 	adrp	x2, 17000 <__FRAME_END__+0xf8b8>
    14c8:	f947f042 	ldr	x2, [x2, #4064]
    14cc:	b4000062 	cbz	x2, 14d8 <register_tm_clones+0x38>
    14d0:	aa0203f0 	mov	x16, x2
    14d4:	d61f0200 	br	x16
    14d8:	d65f03c0 	ret
    14dc:	d503201f 	nop

00000000000014e0 <__do_global_dtors_aux>:
    14e0:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    14e4:	910003fd 	mov	x29, sp
    14e8:	f9000bf3 	str	x19, [sp, #16]
    14ec:	f00000b3 	adrp	x19, 18000 <memcpy@GLIBC_2.17>
    14f0:	394b8260 	ldrb	w0, [x19, #736]
    14f4:	35000140 	cbnz	w0, 151c <__do_global_dtors_aux+0x3c>
    14f8:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    14fc:	f947d400 	ldr	x0, [x0, #4008]
    1500:	b4000080 	cbz	x0, 1510 <__do_global_dtors_aux+0x30>
    1504:	f00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    1508:	f940a800 	ldr	x0, [x0, #336]
    150c:	97ffff21 	bl	1190 <__cxa_finalize@plt>
    1510:	97ffffd8 	bl	1470 <deregister_tm_clones>
    1514:	52800020 	mov	w0, #0x1                   	// #1
    1518:	390b8260 	strb	w0, [x19, #736]
    151c:	f9400bf3 	ldr	x19, [sp, #16]
    1520:	a8c27bfd 	ldp	x29, x30, [sp], #32
    1524:	d65f03c0 	ret
    1528:	d503201f 	nop
    152c:	d503201f 	nop

0000000000001530 <frame_dummy>:
    1530:	17ffffdc 	b	14a0 <register_tm_clones>

0000000000001534 <print_usage>:
#include "helper_funcs.h"
#include "types.h"
#include "sptensors.h"
#include "matricies.h"

static void print_usage(char ** argv) {
    1534:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1538:	910003fd 	mov	x29, sp
    153c:	f9000fe0 	str	x0, [sp, #24]
	printf("Usage: %s [options] \n\n", argv[0]);
    1540:	f9400fe0 	ldr	x0, [sp, #24]
    1544:	f9400000 	ldr	x0, [x0]
    1548:	aa0003e1 	mov	x1, x0
    154c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1550:	910e8000 	add	x0, x0, #0x3a0
    1554:	97ffff97 	bl	13b0 <printf@plt>
	printf("Options: -i INPUT, --input=INPUT (.tns file)\n");
    1558:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    155c:	910ee000 	add	x0, x0, #0x3b8
    1560:	97ffff6c 	bl	1310 <puts@plt>
	printf("         -o OUTPUT, --output=OUTPUT (output file name)\n");
    1564:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1568:	910fa000 	add	x0, x0, #0x3e8
    156c:	97ffff69 	bl	1310 <puts@plt>
	printf("         -m MODE, --mode=MODE (specify a mode, e.g., 0 (default) or 1 or 2 for third-order tensors.)\n");
    1570:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1574:	91108000 	add	x0, x0, #0x420
    1578:	97ffff66 	bl	1310 <puts@plt>
	printf("         -d DEV_ID, --dev-id=DEV_ID (-2:sequential,default; -1:OpenMP parallel)\n");
    157c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1580:	91122000 	add	x0, x0, #0x488
    1584:	97ffff63 	bl	1310 <puts@plt>
	printf("         -r RANK (the number of matrix columns, 16:default)\n");
    1588:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    158c:	91136000 	add	x0, x0, #0x4d8
    1590:	97ffff60 	bl	1310 <puts@plt>
	printf("         -v VALIDATION, --validate=VALIDFILE (a previous output file to compare against). This also removes randomisation from matrix creation\n");
    1594:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1598:	91146000 	add	x0, x0, #0x518
    159c:	97ffff5d 	bl	1310 <puts@plt>
	printf("         --help\n");
    15a0:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    15a4:	9116a000 	add	x0, x0, #0x5a8
    15a8:	97ffff5a 	bl	1310 <puts@plt>
	printf("\n");
    15ac:	52800140 	mov	w0, #0xa                   	// #10
    15b0:	97ffff8c 	bl	13e0 <putchar@plt>
}
    15b4:	d503201f 	nop
    15b8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    15bc:	d65f03c0 	ret

00000000000015c0 <main>:

/**
 * Benchmark Matriced Tensor Times Khatri-Rao Product (MTTKRP), tensor in COO format, matrices are dense.
 */
int main(int argc, char ** argv)
{
    15c0:	d13383ff 	sub	sp, sp, #0xce0
    15c4:	a9007bfd 	stp	x29, x30, [sp]
    15c8:	910003fd 	mov	x29, sp
    15cc:	f9000bf3 	str	x19, [sp, #16]
    15d0:	b9002fe0 	str	w0, [sp, #44]
    15d4:	f90013e1 	str	x1, [sp, #32]
	FILE *fo = NULL;
    15d8:	f9066fff 	str	xzr, [sp, #3288]
	char fvname[1000];
	char foname[1000];
	sptSparseTensor X;
	sptMatrix ** U;

	bool random = true;
    15dc:	52800020 	mov	w0, #0x1                   	// #1
    15e0:	39335fe0 	strb	w0, [sp, #3287]
	sptIndex mode = 0;
    15e4:	b90057ff 	str	wzr, [sp, #84]
	sptIndex R = 16;
    15e8:	52800200 	mov	w0, #0x10                  	// #16
    15ec:	b90053e0 	str	w0, [sp, #80]
	int dev_id = -2;
    15f0:	12800020 	mov	w0, #0xfffffffe            	// #-2
    15f4:	b9004fe0 	str	w0, [sp, #76]
	int niters = 5;
    15f8:	528000a0 	mov	w0, #0x5                   	// #5
    15fc:	b90ca7e0 	str	w0, [sp, #3236]
	int nthreads = 1;
    1600:	52800020 	mov	w0, #0x1                   	// #1
    1604:	b90cd3e0 	str	w0, [sp, #3280]
	printf("niters: %d\n", niters);
    1608:	b94ca7e1 	ldr	w1, [sp, #3236]
    160c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1610:	9116e000 	add	x0, x0, #0x5b8
    1614:	97ffff67 	bl	13b0 <printf@plt>

	if(argc <= 3) { // #Required arguments
    1618:	b9402fe0 	ldr	w0, [sp, #44]
    161c:	71000c1f 	cmp	w0, #0x3
    1620:	540000ac 	b.gt	1634 <main+0x74>
		print_usage(argv);
    1624:	f94013e0 	ldr	x0, [sp, #32]
    1628:	97ffffc3 	bl	1534 <print_usage>
		exit(1);
    162c:	52800020 	mov	w0, #0x1                   	// #1
    1630:	97fffed4 	bl	1180 <exit@plt>
			{"validate", optional_argument, 0, 'v'},
			{0, 0, 0, 0}
	};
	int c;
	for(;;) {
		int option_index = 0;
    1634:	b9003fff 	str	wzr, [sp, #60]
		c = getopt_long(argc, argv, "i:m:o:d:r:v:", long_options, &option_index);
    1638:	9100f3e0 	add	x0, sp, #0x3c
    163c:	aa0003e4 	mov	x4, x0
    1640:	f00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    1644:	91056003 	add	x3, x0, #0x158
    1648:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    164c:	91172002 	add	x2, x0, #0x5c8
    1650:	f94013e1 	ldr	x1, [sp, #32]
    1654:	b9402fe0 	ldr	w0, [sp, #44]
    1658:	97ffff32 	bl	1320 <getopt_long@plt>
    165c:	b90ca3e0 	str	w0, [sp, #3232]
		if(c == -1) {
    1660:	b94ca3e0 	ldr	w0, [sp, #3232]
    1664:	3100041f 	cmn	w0, #0x1
    1668:	54000261 	b.ne	16b4 <main+0xf4>  // b.any
				print_usage(argv);
				exit(1);
		}
	}

	printf("mode: %"PASTA_PRI_INDEX "\n", mode);
    166c:	b94057e0 	ldr	w0, [sp, #84]
    1670:	2a0003e1 	mov	w1, w0
    1674:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1678:	91176000 	add	x0, x0, #0x5d8
    167c:	97ffff4d 	bl	13b0 <printf@plt>
	printf("dev_id: %d\n", dev_id);
    1680:	b9404fe0 	ldr	w0, [sp, #76]
    1684:	2a0003e1 	mov	w1, w0
    1688:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    168c:	9117a000 	add	x0, x0, #0x5e8
    1690:	97ffff48 	bl	13b0 <printf@plt>

	/* Load a sparse tensor from file as it is */
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    1694:	9121a3e1 	add	x1, sp, #0x868
    1698:	910163e0 	add	x0, sp, #0x58
    169c:	aa0103e2 	mov	x2, x1
    16a0:	52800021 	mov	w1, #0x1                   	// #1
    16a4:	94000857 	bl	3800 <sptLoadSparseTensor>
    16a8:	7100001f 	cmp	w0, #0x0
    16ac:	54001221 	b.ne	18f0 <main+0x330>  // b.any
    16b0:	14000092 	b	18f8 <main+0x338>
		switch(c) {
    16b4:	b94ca3e0 	ldr	w0, [sp, #3232]
    16b8:	7101d81f 	cmp	w0, #0x76
    16bc:	54000ee0 	b.eq	1898 <main+0x2d8>  // b.none
    16c0:	b94ca3e0 	ldr	w0, [sp, #3232]
    16c4:	7101d81f 	cmp	w0, #0x76
    16c8:	5400108c 	b.gt	18d8 <main+0x318>
    16cc:	b94ca3e0 	ldr	w0, [sp, #3232]
    16d0:	7101c81f 	cmp	w0, #0x72
    16d4:	54000ce0 	b.eq	1870 <main+0x2b0>  // b.none
    16d8:	b94ca3e0 	ldr	w0, [sp, #3232]
    16dc:	7101c81f 	cmp	w0, #0x72
    16e0:	54000fcc 	b.gt	18d8 <main+0x318>
    16e4:	b94ca3e0 	ldr	w0, [sp, #3232]
    16e8:	7101bc1f 	cmp	w0, #0x6f
    16ec:	540003e0 	b.eq	1768 <main+0x1a8>  // b.none
    16f0:	b94ca3e0 	ldr	w0, [sp, #3232]
    16f4:	7101bc1f 	cmp	w0, #0x6f
    16f8:	54000f0c 	b.gt	18d8 <main+0x318>
    16fc:	b94ca3e0 	ldr	w0, [sp, #3232]
    1700:	7101b41f 	cmp	w0, #0x6d
    1704:	540006e0 	b.eq	17e0 <main+0x220>  // b.none
    1708:	b94ca3e0 	ldr	w0, [sp, #3232]
    170c:	7101b41f 	cmp	w0, #0x6d
    1710:	54000e4c 	b.gt	18d8 <main+0x318>
    1714:	b94ca3e0 	ldr	w0, [sp, #3232]
    1718:	7101901f 	cmp	w0, #0x64
    171c:	54000760 	b.eq	1808 <main+0x248>  // b.none
    1720:	b94ca3e0 	ldr	w0, [sp, #3232]
    1724:	7101a41f 	cmp	w0, #0x69
    1728:	54000d81 	b.ne	18d8 <main+0x318>  // b.any
				strcpy(fname, optarg);
    172c:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    1730:	f947dc00 	ldr	x0, [x0, #4024]
    1734:	f9400001 	ldr	x1, [x0]
    1738:	9121a3e0 	add	x0, sp, #0x868
    173c:	97ffff11 	bl	1380 <strcpy@plt>
				printf("input file: %s\n", fname); fflush(stdout);
    1740:	9121a3e0 	add	x0, sp, #0x868
    1744:	aa0003e1 	mov	x1, x0
    1748:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    174c:	9117e000 	add	x0, x0, #0x5f8
    1750:	97ffff18 	bl	13b0 <printf@plt>
    1754:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    1758:	f947e000 	ldr	x0, [x0, #4032]
    175c:	f9400000 	ldr	x0, [x0]
    1760:	97ffff04 	bl	1370 <fflush@plt>
				break;
    1764:	14000062 	b	18ec <main+0x32c>
				fo = fopen(optarg, "w");
    1768:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    176c:	f947dc00 	ldr	x0, [x0, #4024]
    1770:	f9400002 	ldr	x2, [x0]
    1774:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1778:	91182001 	add	x1, x0, #0x608
    177c:	aa0203e0 	mov	x0, x2
    1780:	97fffea0 	bl	1200 <fopen@plt>
    1784:	f9066fe0 	str	x0, [sp, #3288]
				strcpy(foname, optarg);
    1788:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    178c:	f947dc00 	ldr	x0, [x0, #4024]
    1790:	f9400001 	ldr	x1, [x0]
    1794:	910263e0 	add	x0, sp, #0x98
    1798:	97fffefa 	bl	1380 <strcpy@plt>
				sptAssert(fo != NULL);
    179c:	f9466fe0 	ldr	x0, [sp, #3288]
    17a0:	f100001f 	cmp	x0, #0x0
    17a4:	54000061 	b.ne	17b0 <main+0x1f0>  // b.any
    17a8:	12800000 	mov	w0, #0xffffffff            	// #-1
    17ac:	97fffe75 	bl	1180 <exit@plt>
				printf("output file: %s\n", optarg); fflush(stdout);
    17b0:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    17b4:	f947dc00 	ldr	x0, [x0, #4024]
    17b8:	f9400000 	ldr	x0, [x0]
    17bc:	aa0003e1 	mov	x1, x0
    17c0:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    17c4:	91184000 	add	x0, x0, #0x610
    17c8:	97fffefa 	bl	13b0 <printf@plt>
    17cc:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    17d0:	f947e000 	ldr	x0, [x0, #4032]
    17d4:	f9400000 	ldr	x0, [x0]
    17d8:	97fffee6 	bl	1370 <fflush@plt>
				break;
    17dc:	14000044 	b	18ec <main+0x32c>
				sscanf(optarg, "%"PASTA_SCN_INDEX, &mode);
    17e0:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    17e4:	f947dc00 	ldr	x0, [x0, #4024]
    17e8:	f9400003 	ldr	x3, [x0]
    17ec:	910153e0 	add	x0, sp, #0x54
    17f0:	aa0003e2 	mov	x2, x0
    17f4:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    17f8:	9118a001 	add	x1, x0, #0x628
    17fc:	aa0303e0 	mov	x0, x3
    1800:	97fffee8 	bl	13a0 <__isoc99_sscanf@plt>
				break;
    1804:	1400003a 	b	18ec <main+0x32c>
				sscanf(optarg, "%d", &dev_id);
    1808:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    180c:	f947dc00 	ldr	x0, [x0, #4024]
    1810:	f9400003 	ldr	x3, [x0]
    1814:	910133e0 	add	x0, sp, #0x4c
    1818:	aa0003e2 	mov	x2, x0
    181c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1820:	9118c001 	add	x1, x0, #0x630
    1824:	aa0303e0 	mov	x0, x3
    1828:	97fffede 	bl	13a0 <__isoc99_sscanf@plt>
				if(dev_id < -2 || dev_id >= 0) {
    182c:	b9404fe0 	ldr	w0, [sp, #76]
    1830:	3100081f 	cmn	w0, #0x2
    1834:	5400008b 	b.lt	1844 <main+0x284>  // b.tstop
    1838:	b9404fe0 	ldr	w0, [sp, #76]
    183c:	7100001f 	cmp	w0, #0x0
    1840:	5400054b 	b.lt	18e8 <main+0x328>  // b.tstop
					fprintf(stderr, "Error: set dev_id to -2/-1.\n");
    1844:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    1848:	f947d800 	ldr	x0, [x0, #4016]
    184c:	f9400000 	ldr	x0, [x0]
    1850:	aa0003e3 	mov	x3, x0
    1854:	d2800382 	mov	x2, #0x1c                  	// #28
    1858:	d2800021 	mov	x1, #0x1                   	// #1
    185c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1860:	9118e000 	add	x0, x0, #0x638
    1864:	97fffebf 	bl	1360 <fwrite@plt>
					exit(1);
    1868:	52800020 	mov	w0, #0x1                   	// #1
    186c:	97fffe45 	bl	1180 <exit@plt>
				sscanf(optarg, "%u"PASTA_SCN_INDEX, &R);
    1870:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    1874:	f947dc00 	ldr	x0, [x0, #4024]
    1878:	f9400003 	ldr	x3, [x0]
    187c:	910143e0 	add	x0, sp, #0x50
    1880:	aa0003e2 	mov	x2, x0
    1884:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1888:	91196001 	add	x1, x0, #0x658
    188c:	aa0303e0 	mov	x0, x3
    1890:	97fffec4 	bl	13a0 <__isoc99_sscanf@plt>
				break;
    1894:	14000016 	b	18ec <main+0x32c>
				random = false;
    1898:	39335fff 	strb	wzr, [sp, #3287]
				strcpy(fvname, optarg);
    189c:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    18a0:	f947dc00 	ldr	x0, [x0, #4024]
    18a4:	f9400001 	ldr	x1, [x0]
    18a8:	911203e0 	add	x0, sp, #0x480
    18ac:	97fffeb5 	bl	1380 <strcpy@plt>
				printf("validation input file: %s\n", fvname); fflush(stdout);
    18b0:	911203e0 	add	x0, sp, #0x480
    18b4:	aa0003e1 	mov	x1, x0
    18b8:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    18bc:	91198000 	add	x0, x0, #0x660
    18c0:	97fffebc 	bl	13b0 <printf@plt>
    18c4:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    18c8:	f947e000 	ldr	x0, [x0, #4032]
    18cc:	f9400000 	ldr	x0, [x0]
    18d0:	97fffea8 	bl	1370 <fflush@plt>
				break;
    18d4:	14000006 	b	18ec <main+0x32c>
				print_usage(argv);
    18d8:	f94013e0 	ldr	x0, [sp, #32]
    18dc:	97ffff16 	bl	1534 <print_usage>
				exit(1);
    18e0:	52800020 	mov	w0, #0x1                   	// #1
    18e4:	97fffe27 	bl	1180 <exit@plt>
				break;
    18e8:	d503201f 	nop
	for(;;) {
    18ec:	17ffff52 	b	1634 <main+0x74>
	sptAssert(sptLoadSparseTensor(&X, 1, fname) == 0);
    18f0:	12800000 	mov	w0, #0xffffffff            	// #-1
    18f4:	97fffe23 	bl	1180 <exit@plt>
	sptSparseTensorStatus(&X, stdout);
    18f8:	d00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    18fc:	f947e000 	ldr	x0, [x0, #4032]
    1900:	f9400001 	ldr	x1, [x0]
    1904:	910163e0 	add	x0, sp, #0x58
    1908:	940009ea 	bl	40b0 <sptSparseTensorStatus>

	sptIndex nmodes = X.nmodes;
    190c:	b9405be0 	ldr	w0, [sp, #88]
    1910:	b90c9fe0 	str	w0, [sp, #3228]
	U = (sptMatrix **)malloc((nmodes+1) * sizeof(sptMatrix*));
    1914:	b94c9fe0 	ldr	w0, [sp, #3228]
    1918:	11000400 	add	w0, w0, #0x1
    191c:	2a0003e0 	mov	w0, w0
    1920:	d37df000 	lsl	x0, x0, #3
    1924:	97fffe3f 	bl	1220 <malloc@plt>
    1928:	f9064be0 	str	x0, [sp, #3216]
	for(sptIndex m=0; m<nmodes+1; ++m) {
    192c:	b90ccfff 	str	wzr, [sp, #3276]
    1930:	1400000b 	b	195c <main+0x39c>
		U[m] = (sptMatrix *)malloc(sizeof(sptMatrix));
    1934:	b94ccfe0 	ldr	w0, [sp, #3276]
    1938:	d37df000 	lsl	x0, x0, #3
    193c:	f9464be1 	ldr	x1, [sp, #3216]
    1940:	8b000033 	add	x19, x1, x0
    1944:	d2800300 	mov	x0, #0x18                  	// #24
    1948:	97fffe36 	bl	1220 <malloc@plt>
    194c:	f9000260 	str	x0, [x19]
	for(sptIndex m=0; m<nmodes+1; ++m) {
    1950:	b94ccfe0 	ldr	w0, [sp, #3276]
    1954:	11000400 	add	w0, w0, #0x1
    1958:	b90ccfe0 	str	w0, [sp, #3276]
    195c:	b94c9fe0 	ldr	w0, [sp, #3228]
    1960:	11000400 	add	w0, w0, #0x1
    1964:	b94ccfe1 	ldr	w1, [sp, #3276]
    1968:	6b00003f 	cmp	w1, w0
    196c:	54fffe43 	b.cc	1934 <main+0x374>  // b.lo, b.ul, b.last
	}
	sptIndex max_ndims = 0;
    1970:	b90ccbff 	str	wzr, [sp, #3272]
	for(sptIndex m=0; m<nmodes; ++m) {
    1974:	b90cc7ff 	str	wzr, [sp, #3268]
    1978:	14000030 	b	1a38 <main+0x478>
		sptAssert(sptNewMatrix(U[m], X.ndims[m], R) == 0);
    197c:	b94cc7e0 	ldr	w0, [sp, #3268]
    1980:	d37df000 	lsl	x0, x0, #3
    1984:	f9464be1 	ldr	x1, [sp, #3216]
    1988:	8b000020 	add	x0, x1, x0
    198c:	f9400003 	ldr	x3, [x0]
    1990:	f94037e1 	ldr	x1, [sp, #104]
    1994:	b94cc7e0 	ldr	w0, [sp, #3268]
    1998:	d37ef400 	lsl	x0, x0, #2
    199c:	8b000020 	add	x0, x1, x0
    19a0:	b9400000 	ldr	w0, [x0]
    19a4:	b94053e1 	ldr	w1, [sp, #80]
    19a8:	2a0103e2 	mov	w2, w1
    19ac:	2a0003e1 	mov	w1, w0
    19b0:	aa0303e0 	mov	x0, x3
    19b4:	940008ba 	bl	3c9c <sptNewMatrix>
    19b8:	7100001f 	cmp	w0, #0x0
    19bc:	54000060 	b.eq	19c8 <main+0x408>  // b.none
    19c0:	12800000 	mov	w0, #0xffffffff            	// #-1
    19c4:	97fffdef 	bl	1180 <exit@plt>
		// sptAssert(sptConstantMatrix(U[m], 1) == 0);
		sptAssert(sptRandomizeMatrix(U[m], random) == 0);
    19c8:	b94cc7e0 	ldr	w0, [sp, #3268]
    19cc:	d37df000 	lsl	x0, x0, #3
    19d0:	f9464be1 	ldr	x1, [sp, #3216]
    19d4:	8b000020 	add	x0, x1, x0
    19d8:	f9400000 	ldr	x0, [x0]
    19dc:	39735fe1 	ldrb	w1, [sp, #3287]
    19e0:	9400090a 	bl	3e08 <sptRandomizeMatrix>
    19e4:	7100001f 	cmp	w0, #0x0
    19e8:	54000060 	b.eq	19f4 <main+0x434>  // b.none
    19ec:	12800000 	mov	w0, #0xffffffff            	// #-1
    19f0:	97fffde4 	bl	1180 <exit@plt>
		if(X.ndims[m] > max_ndims)
    19f4:	f94037e1 	ldr	x1, [sp, #104]
    19f8:	b94cc7e0 	ldr	w0, [sp, #3268]
    19fc:	d37ef400 	lsl	x0, x0, #2
    1a00:	8b000020 	add	x0, x1, x0
    1a04:	b9400000 	ldr	w0, [x0]
    1a08:	b94ccbe1 	ldr	w1, [sp, #3272]
    1a0c:	6b00003f 	cmp	w1, w0
    1a10:	540000e2 	b.cs	1a2c <main+0x46c>  // b.hs, b.nlast
			max_ndims = X.ndims[m];
    1a14:	f94037e1 	ldr	x1, [sp, #104]
    1a18:	b94cc7e0 	ldr	w0, [sp, #3268]
    1a1c:	d37ef400 	lsl	x0, x0, #2
    1a20:	8b000020 	add	x0, x1, x0
    1a24:	b9400000 	ldr	w0, [x0]
    1a28:	b90ccbe0 	str	w0, [sp, #3272]
	for(sptIndex m=0; m<nmodes; ++m) {
    1a2c:	b94cc7e0 	ldr	w0, [sp, #3268]
    1a30:	11000400 	add	w0, w0, #0x1
    1a34:	b90cc7e0 	str	w0, [sp, #3268]
    1a38:	b94cc7e1 	ldr	w1, [sp, #3268]
    1a3c:	b94c9fe0 	ldr	w0, [sp, #3228]
    1a40:	6b00003f 	cmp	w1, w0
    1a44:	54fff9c3 	b.cc	197c <main+0x3bc>  // b.lo, b.ul, b.last
	}
	sptAssert(sptNewMatrix(U[nmodes], max_ndims, R) == 0);
    1a48:	b94c9fe0 	ldr	w0, [sp, #3228]
    1a4c:	d37df000 	lsl	x0, x0, #3
    1a50:	f9464be1 	ldr	x1, [sp, #3216]
    1a54:	8b000020 	add	x0, x1, x0
    1a58:	f9400000 	ldr	x0, [x0]
    1a5c:	b94053e1 	ldr	w1, [sp, #80]
    1a60:	2a0103e2 	mov	w2, w1
    1a64:	b94ccbe1 	ldr	w1, [sp, #3272]
    1a68:	9400088d 	bl	3c9c <sptNewMatrix>
    1a6c:	7100001f 	cmp	w0, #0x0
    1a70:	54000060 	b.eq	1a7c <main+0x4bc>  // b.none
    1a74:	12800000 	mov	w0, #0xffffffff            	// #-1
    1a78:	97fffdc2 	bl	1180 <exit@plt>
	sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1a7c:	b94c9fe0 	ldr	w0, [sp, #3228]
    1a80:	d37df000 	lsl	x0, x0, #3
    1a84:	f9464be1 	ldr	x1, [sp, #3216]
    1a88:	8b000020 	add	x0, x1, x0
    1a8c:	f9400000 	ldr	x0, [x0]
    1a90:	0f000400 	movi	v0.2s, #0x0
    1a94:	94000925 	bl	3f28 <sptConstantMatrix>
    1a98:	7100001f 	cmp	w0, #0x0
    1a9c:	54000060 	b.eq	1aa8 <main+0x4e8>  // b.none
    1aa0:	12800000 	mov	w0, #0xffffffff            	// #-1
    1aa4:	97fffdb7 	bl	1180 <exit@plt>
	sptIndex stride = U[0]->stride;
    1aa8:	f9464be0 	ldr	x0, [sp, #3216]
    1aac:	f9400000 	ldr	x0, [x0]
    1ab0:	b9400c00 	ldr	w0, [x0, #12]
    1ab4:	b90c8fe0 	str	w0, [sp, #3212]

	sptIndex * mats_order = (sptIndex*)malloc(nmodes * sizeof(sptIndex));
    1ab8:	b94c9fe0 	ldr	w0, [sp, #3228]
    1abc:	d37ef400 	lsl	x0, x0, #2
    1ac0:	97fffdd8 	bl	1220 <malloc@plt>
    1ac4:	f90643e0 	str	x0, [sp, #3200]
	mats_order[0] = mode;
    1ac8:	b94057e1 	ldr	w1, [sp, #84]
    1acc:	f94643e0 	ldr	x0, [sp, #3200]
    1ad0:	b9000001 	str	w1, [x0]
	for(sptIndex i=1; i<nmodes; ++i)
    1ad4:	52800020 	mov	w0, #0x1                   	// #1
    1ad8:	b90cc3e0 	str	w0, [sp, #3264]
    1adc:	14000011 	b	1b20 <main+0x560>
		mats_order[i] = (mode+i) % nmodes;
    1ae0:	b94057e1 	ldr	w1, [sp, #84]
    1ae4:	b94cc3e0 	ldr	w0, [sp, #3264]
    1ae8:	0b000020 	add	w0, w1, w0
    1aec:	b94cc3e1 	ldr	w1, [sp, #3264]
    1af0:	d37ef421 	lsl	x1, x1, #2
    1af4:	f94643e2 	ldr	x2, [sp, #3200]
    1af8:	8b010041 	add	x1, x2, x1
    1afc:	b94c9fe2 	ldr	w2, [sp, #3228]
    1b00:	1ac20803 	udiv	w3, w0, w2
    1b04:	b94c9fe2 	ldr	w2, [sp, #3228]
    1b08:	1b027c62 	mul	w2, w3, w2
    1b0c:	4b020000 	sub	w0, w0, w2
    1b10:	b9000020 	str	w0, [x1]
	for(sptIndex i=1; i<nmodes; ++i)
    1b14:	b94cc3e0 	ldr	w0, [sp, #3264]
    1b18:	11000400 	add	w0, w0, #0x1
    1b1c:	b90cc3e0 	str	w0, [sp, #3264]
    1b20:	b94cc3e1 	ldr	w1, [sp, #3264]
    1b24:	b94c9fe0 	ldr	w0, [sp, #3228]
    1b28:	6b00003f 	cmp	w1, w0
    1b2c:	54fffda3 	b.cc	1ae0 <main+0x520>  // b.lo, b.ul, b.last

	/* For warm-up caches, timing not included */
	if(dev_id == -2) {
    1b30:	b9404fe0 	ldr	w0, [sp, #76]
    1b34:	3100081f 	cmn	w0, #0x2
    1b38:	540001a1 	b.ne	1b6c <main+0x5ac>  // b.any
		nthreads = 1;
    1b3c:	52800020 	mov	w0, #0x1                   	// #1
    1b40:	b90cd3e0 	str	w0, [sp, #3280]
		sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1b44:	b94057e1 	ldr	w1, [sp, #84]
    1b48:	910163e0 	add	x0, sp, #0x58
    1b4c:	2a0103e3 	mov	w3, w1
    1b50:	f94643e2 	ldr	x2, [sp, #3200]
    1b54:	f9464be1 	ldr	x1, [sp, #3216]
    1b58:	940009d4 	bl	42a8 <sptMTTKRP>
    1b5c:	7100001f 	cmp	w0, #0x0
    1b60:	54000400 	b.eq	1be0 <main+0x620>  // b.none
    1b64:	12800000 	mov	w0, #0xffffffff            	// #-1
    1b68:	97fffd86 	bl	1180 <exit@plt>
	} else if(dev_id == -1) {
    1b6c:	b9404fe0 	ldr	w0, [sp, #76]
    1b70:	3100041f 	cmn	w0, #0x1
    1b74:	54000361 	b.ne	1be0 <main+0x620>  // b.any
#ifdef PASTA_USE_OPENMP
		#pragma omp parallel
    1b78:	b94cd3e0 	ldr	w0, [sp, #3280]
    1b7c:	b90c53e0 	str	w0, [sp, #3152]
    1b80:	913143e0 	add	x0, sp, #0xc50
    1b84:	52800003 	mov	w3, #0x0                   	// #0
    1b88:	52800002 	mov	w2, #0x0                   	// #0
    1b8c:	aa0003e1 	mov	x1, x0
    1b90:	90000000 	adrp	x0, 1000 <g_last_error+0x1000>
    1b94:	913fa000 	add	x0, x0, #0xfe8
    1b98:	97fffe0e 	bl	13d0 <GOMP_parallel@plt>
    1b9c:	b94c53e0 	ldr	w0, [sp, #3152]
    1ba0:	b90cd3e0 	str	w0, [sp, #3280]
        {
            nthreads = omp_get_num_threads();
        }
        printf("\nnthreads: %d\n", nthreads);
    1ba4:	b94cd3e1 	ldr	w1, [sp, #3280]
    1ba8:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1bac:	911a0000 	add	x0, x0, #0x680
    1bb0:	97fffe00 	bl	13b0 <printf@plt>
        sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1bb4:	b94057e1 	ldr	w1, [sp, #84]
    1bb8:	910163e0 	add	x0, sp, #0x58
    1bbc:	b94cd3e4 	ldr	w4, [sp, #3280]
    1bc0:	2a0103e3 	mov	w3, w1
    1bc4:	f94643e2 	ldr	x2, [sp, #3200]
    1bc8:	f9464be1 	ldr	x1, [sp, #3216]
    1bcc:	94000c36 	bl	4ca4 <sptOmpMTTKRP>
    1bd0:	7100001f 	cmp	w0, #0x0
    1bd4:	54000060 	b.eq	1be0 <main+0x620>  // b.none
    1bd8:	12800000 	mov	w0, #0xffffffff            	// #-1
    1bdc:	97fffd69 	bl	1180 <exit@plt>
#endif
	}


	sptTimer timer;
	sptNewTimer(&timer, 0);
    1be0:	910103e0 	add	x0, sp, #0x40
    1be4:	52800001 	mov	w1, #0x0                   	// #0
    1be8:	94000f5d 	bl	595c <sptNewTimer>
	sptStartTimer(timer);
    1bec:	f94023e0 	ldr	x0, [sp, #64]
    1bf0:	94000f7b 	bl	59dc <sptStartTimer>

	for(int it=0; it<niters; ++it) {
    1bf4:	b90cbfff 	str	wzr, [sp, #3260]
    1bf8:	1400002a 	b	1ca0 <main+0x6e0>
		sptAssert(sptConstantMatrix(U[nmodes], 0) == 0);
    1bfc:	b94c9fe0 	ldr	w0, [sp, #3228]
    1c00:	d37df000 	lsl	x0, x0, #3
    1c04:	f9464be1 	ldr	x1, [sp, #3216]
    1c08:	8b000020 	add	x0, x1, x0
    1c0c:	f9400000 	ldr	x0, [x0]
    1c10:	0f000400 	movi	v0.2s, #0x0
    1c14:	940008c5 	bl	3f28 <sptConstantMatrix>
    1c18:	7100001f 	cmp	w0, #0x0
    1c1c:	54000060 	b.eq	1c28 <main+0x668>  // b.none
    1c20:	12800000 	mov	w0, #0xffffffff            	// #-1
    1c24:	97fffd57 	bl	1180 <exit@plt>
		if(dev_id == -2) {
    1c28:	b9404fe0 	ldr	w0, [sp, #76]
    1c2c:	3100081f 	cmn	w0, #0x2
    1c30:	54000161 	b.ne	1c5c <main+0x69c>  // b.any
			sptAssert(sptMTTKRP(&X, U, mats_order, mode) == 0);
    1c34:	b94057e1 	ldr	w1, [sp, #84]
    1c38:	910163e0 	add	x0, sp, #0x58
    1c3c:	2a0103e3 	mov	w3, w1
    1c40:	f94643e2 	ldr	x2, [sp, #3200]
    1c44:	f9464be1 	ldr	x1, [sp, #3216]
    1c48:	94000998 	bl	42a8 <sptMTTKRP>
    1c4c:	7100001f 	cmp	w0, #0x0
    1c50:	54000220 	b.eq	1c94 <main+0x6d4>  // b.none
    1c54:	12800000 	mov	w0, #0xffffffff            	// #-1
    1c58:	97fffd4a 	bl	1180 <exit@plt>
		} else if(dev_id == -1) {
    1c5c:	b9404fe0 	ldr	w0, [sp, #76]
    1c60:	3100041f 	cmn	w0, #0x1
    1c64:	54000181 	b.ne	1c94 <main+0x6d4>  // b.any
#ifdef PASTA_USE_OPENMP
			sptAssert(sptOmpMTTKRP(&X, U, mats_order, mode, nthreads) == 0);
    1c68:	b94057e1 	ldr	w1, [sp, #84]
    1c6c:	910163e0 	add	x0, sp, #0x58
    1c70:	b94cd3e4 	ldr	w4, [sp, #3280]
    1c74:	2a0103e3 	mov	w3, w1
    1c78:	f94643e2 	ldr	x2, [sp, #3200]
    1c7c:	f9464be1 	ldr	x1, [sp, #3216]
    1c80:	94000c09 	bl	4ca4 <sptOmpMTTKRP>
    1c84:	7100001f 	cmp	w0, #0x0
    1c88:	54000060 	b.eq	1c94 <main+0x6d4>  // b.none
    1c8c:	12800000 	mov	w0, #0xffffffff            	// #-1
    1c90:	97fffd3c 	bl	1180 <exit@plt>
	for(int it=0; it<niters; ++it) {
    1c94:	b94cbfe0 	ldr	w0, [sp, #3260]
    1c98:	11000400 	add	w0, w0, #0x1
    1c9c:	b90cbfe0 	str	w0, [sp, #3260]
    1ca0:	b94cbfe1 	ldr	w1, [sp, #3260]
    1ca4:	b94ca7e0 	ldr	w0, [sp, #3236]
    1ca8:	6b00003f 	cmp	w1, w0
    1cac:	54fffa8b 	b.lt	1bfc <main+0x63c>  // b.tstop
#endif
		}
	}

	sptStopTimer(timer);
    1cb0:	f94023e0 	ldr	x0, [sp, #64]
    1cb4:	94000f8a 	bl	5adc <sptStopTimer>

	double aver_time = sptPrintAverageElapsedTime(timer, niters, "Average CooMTTKRP");
    1cb8:	f94023e3 	ldr	x3, [sp, #64]
    1cbc:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1cc0:	911a4002 	add	x2, x0, #0x690
    1cc4:	b94ca7e1 	ldr	w1, [sp, #3236]
    1cc8:	aa0303e0 	mov	x0, x3
    1ccc:	94000ff4 	bl	5c9c <sptPrintAverageElapsedTime>
    1cd0:	fd063fe0 	str	d0, [sp, #3192]
	double gflops = (double)nmodes * R * X.nnz / aver_time / 1e9;
    1cd4:	b94c9fe0 	ldr	w0, [sp, #3228]
    1cd8:	1e630001 	ucvtf	d1, w0
    1cdc:	b94053e0 	ldr	w0, [sp, #80]
    1ce0:	1e630000 	ucvtf	d0, w0
    1ce4:	1e600821 	fmul	d1, d1, d0
    1ce8:	fd403be0 	ldr	d0, [sp, #112]
    1cec:	7e61d800 	ucvtf	d0, d0
    1cf0:	1e600821 	fmul	d1, d1, d0
    1cf4:	fd463fe0 	ldr	d0, [sp, #3192]
    1cf8:	1e601820 	fdiv	d0, d1, d0
    1cfc:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1d00:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    1d04:	9e670001 	fmov	d1, x0
    1d08:	1e611800 	fdiv	d0, d0, d1
    1d0c:	fd063be0 	str	d0, [sp, #3184]
	uint64_t bytes = ( nmodes * sizeof(sptIndex) + sizeof(sptValue) ) * X.nnz;
    1d10:	b94c9fe0 	ldr	w0, [sp, #3228]
    1d14:	91000401 	add	x1, x0, #0x1
    1d18:	f9403be0 	ldr	x0, [sp, #112]
    1d1c:	9b007c20 	mul	x0, x1, x0
    1d20:	d37ef400 	lsl	x0, x0, #2
    1d24:	f9065be0 	str	x0, [sp, #3248]
	for (sptIndex m=0; m<nmodes; ++m) {
    1d28:	b90cafff 	str	wzr, [sp, #3244]
    1d2c:	14000010 	b	1d6c <main+0x7ac>
		bytes += X.ndims[m] * R * sizeof(sptValue);
    1d30:	f94037e1 	ldr	x1, [sp, #104]
    1d34:	b94cafe0 	ldr	w0, [sp, #3244]
    1d38:	d37ef400 	lsl	x0, x0, #2
    1d3c:	8b000020 	add	x0, x1, x0
    1d40:	b9400001 	ldr	w1, [x0]
    1d44:	b94053e0 	ldr	w0, [sp, #80]
    1d48:	1b007c20 	mul	w0, w1, w0
    1d4c:	2a0003e0 	mov	w0, w0
    1d50:	d37ef400 	lsl	x0, x0, #2
    1d54:	f9465be1 	ldr	x1, [sp, #3248]
    1d58:	8b000020 	add	x0, x1, x0
    1d5c:	f9065be0 	str	x0, [sp, #3248]
	for (sptIndex m=0; m<nmodes; ++m) {
    1d60:	b94cafe0 	ldr	w0, [sp, #3244]
    1d64:	11000400 	add	w0, w0, #0x1
    1d68:	b90cafe0 	str	w0, [sp, #3244]
    1d6c:	b94cafe1 	ldr	w1, [sp, #3244]
    1d70:	b94c9fe0 	ldr	w0, [sp, #3228]
    1d74:	6b00003f 	cmp	w1, w0
    1d78:	54fffdc3 	b.cc	1d30 <main+0x770>  // b.lo, b.ul, b.last
	}
	double gbw = (double)bytes / aver_time / 1e9;
    1d7c:	fd465be0 	ldr	d0, [sp, #3248]
    1d80:	7e61d801 	ucvtf	d1, d0
    1d84:	fd463fe0 	ldr	d0, [sp, #3192]
    1d88:	1e601820 	fdiv	d0, d1, d0
    1d8c:	d2d9aca0 	mov	x0, #0xcd6500000000        	// #225833675390976
    1d90:	f2e839a0 	movk	x0, #0x41cd, lsl #48
    1d94:	9e670001 	fmov	d1, x0
    1d98:	1e611800 	fdiv	d0, d0, d1
    1d9c:	fd0637e0 	str	d0, [sp, #3176]
	printf("Performance: %.2lf GFlop/s, Bandwidth: %.2lf GB/s\n\n", gflops, gbw);
    1da0:	fd4637e1 	ldr	d1, [sp, #3176]
    1da4:	fd463be0 	ldr	d0, [sp, #3184]
    1da8:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1dac:	911aa000 	add	x0, x0, #0x6a8
    1db0:	97fffd80 	bl	13b0 <printf@plt>

	if(fo != NULL) {
    1db4:	f9466fe0 	ldr	x0, [sp, #3288]
    1db8:	f100001f 	cmp	x0, #0x0
    1dbc:	540001c0 	b.eq	1df4 <main+0x834>  // b.none
		sptAssert(sptDumpMatrix(U[nmodes], fo) == 0);
    1dc0:	b94c9fe0 	ldr	w0, [sp, #3228]
    1dc4:	d37df000 	lsl	x0, x0, #3
    1dc8:	f9464be1 	ldr	x1, [sp, #3216]
    1dcc:	8b000020 	add	x0, x1, x0
    1dd0:	f9400000 	ldr	x0, [x0]
    1dd4:	f9466fe1 	ldr	x1, [sp, #3288]
    1dd8:	94000fe2 	bl	5d60 <sptDumpMatrix>
    1ddc:	7100001f 	cmp	w0, #0x0
    1de0:	54000060 	b.eq	1dec <main+0x82c>  // b.none
    1de4:	12800000 	mov	w0, #0xffffffff            	// #-1
    1de8:	97fffce6 	bl	1180 <exit@plt>
		fclose(fo);
    1dec:	f9466fe0 	ldr	x0, [sp, #3288]
    1df0:	97fffd00 	bl	11f0 <fclose@plt>
	}

	sptFreeTimer(timer);
    1df4:	f94023e0 	ldr	x0, [sp, #64]
    1df8:	94000fc1 	bl	5cfc <sptFreeTimer>
	for(sptIndex m=0; m<nmodes; ++m) {
    1dfc:	b90cabff 	str	wzr, [sp, #3240]
    1e00:	1400000a 	b	1e28 <main+0x868>
		sptFreeMatrix(U[m]);
    1e04:	b94cabe0 	ldr	w0, [sp, #3240]
    1e08:	d37df000 	lsl	x0, x0, #3
    1e0c:	f9464be1 	ldr	x1, [sp, #3216]
    1e10:	8b000020 	add	x0, x1, x0
    1e14:	f9400000 	ldr	x0, [x0]
    1e18:	9400086b 	bl	3fc4 <sptFreeMatrix>
	for(sptIndex m=0; m<nmodes; ++m) {
    1e1c:	b94cabe0 	ldr	w0, [sp, #3240]
    1e20:	11000400 	add	w0, w0, #0x1
    1e24:	b90cabe0 	str	w0, [sp, #3240]
    1e28:	b94cabe1 	ldr	w1, [sp, #3240]
    1e2c:	b94c9fe0 	ldr	w0, [sp, #3228]
    1e30:	6b00003f 	cmp	w1, w0
    1e34:	54fffe83 	b.cc	1e04 <main+0x844>  // b.lo, b.ul, b.last
	}
	sptFreeSparseTensor(&X);
    1e38:	910163e0 	add	x0, sp, #0x58
    1e3c:	940000eb 	bl	21e8 <sptFreeSparseTensor>
	free(mats_order);
    1e40:	f94643e0 	ldr	x0, [sp, #3200]
    1e44:	97fffd43 	bl	1350 <free@plt>
	sptFreeMatrix(U[nmodes]);
    1e48:	b94c9fe0 	ldr	w0, [sp, #3228]
    1e4c:	d37df000 	lsl	x0, x0, #3
    1e50:	f9464be1 	ldr	x1, [sp, #3216]
    1e54:	8b000020 	add	x0, x1, x0
    1e58:	f9400000 	ldr	x0, [x0]
    1e5c:	9400085a 	bl	3fc4 <sptFreeMatrix>
	free(U);
    1e60:	f9464be0 	ldr	x0, [sp, #3216]
    1e64:	97fffd3b 	bl	1350 <free@plt>

	if (!random){
    1e68:	39735fe0 	ldrb	w0, [sp, #3287]
    1e6c:	52000000 	eor	w0, w0, #0x1
    1e70:	12001c00 	and	w0, w0, #0xff
    1e74:	7100001f 	cmp	w0, #0x0
    1e78:	540006c0 	b.eq	1f50 <main+0x990>  // b.none
		FILE* fPtr1 = fopen(fvname, "r");
    1e7c:	911203e2 	add	x2, sp, #0x480
    1e80:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1e84:	911b8001 	add	x1, x0, #0x6e0
    1e88:	aa0203e0 	mov	x0, x2
    1e8c:	97fffcdd 	bl	1200 <fopen@plt>
    1e90:	f90633e0 	str	x0, [sp, #3168]
		FILE* fPtr2 = fopen(foname, "r");
    1e94:	910263e2 	add	x2, sp, #0x98
    1e98:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1e9c:	911b8001 	add	x1, x0, #0x6e0
    1ea0:	aa0203e0 	mov	x0, x2
    1ea4:	97fffcd7 	bl	1200 <fopen@plt>
    1ea8:	f9062fe0 	str	x0, [sp, #3160]

		if (fPtr1 == NULL || fPtr2 == NULL) {
    1eac:	f94633e0 	ldr	x0, [sp, #3168]
    1eb0:	f100001f 	cmp	x0, #0x0
    1eb4:	54000080 	b.eq	1ec4 <main+0x904>  // b.none
    1eb8:	f9462fe0 	ldr	x0, [sp, #3160]
    1ebc:	f100001f 	cmp	x0, #0x0
    1ec0:	54000101 	b.ne	1ee0 <main+0x920>  // b.any
			printf("\nUnable to open file.\n");
    1ec4:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1ec8:	911ba000 	add	x0, x0, #0x6e8
    1ecc:	97fffd11 	bl	1310 <puts@plt>
			printf("Please check whether file exists and you have read privilege.\n");
    1ed0:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1ed4:	911c0000 	add	x0, x0, #0x700
    1ed8:	97fffd0e 	bl	1310 <puts@plt>
    1edc:	14000013 	b	1f28 <main+0x968>
		} else {

			int diff = compareFile(fPtr1, fPtr2);
    1ee0:	f9462fe1 	ldr	x1, [sp, #3160]
    1ee4:	f94633e0 	ldr	x0, [sp, #3168]
    1ee8:	9400001f 	bl	1f64 <compareFile>
    1eec:	b90c57e0 	str	w0, [sp, #3156]
			if (diff == 0) {
    1ef0:	b94c57e0 	ldr	w0, [sp, #3156]
    1ef4:	7100001f 	cmp	w0, #0x0
    1ef8:	54000121 	b.ne	1f1c <main+0x95c>  // b.any
				printf("Validation Successful \n %s matchs %s\n", foname, fvname);
    1efc:	911203e1 	add	x1, sp, #0x480
    1f00:	910263e0 	add	x0, sp, #0x98
    1f04:	aa0103e2 	mov	x2, x1
    1f08:	aa0003e1 	mov	x1, x0
    1f0c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1f10:	911d0000 	add	x0, x0, #0x740
    1f14:	97fffd27 	bl	13b0 <printf@plt>
    1f18:	14000004 	b	1f28 <main+0x968>
			} else {
				printf("\nFiles are not equal.\n Validation FAILED \n");
    1f1c:	b0000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    1f20:	911da000 	add	x0, x0, #0x768
    1f24:	97fffcfb 	bl	1310 <puts@plt>
			}
		}
		if(fPtr1 != NULL) fclose(fPtr1);
    1f28:	f94633e0 	ldr	x0, [sp, #3168]
    1f2c:	f100001f 	cmp	x0, #0x0
    1f30:	54000060 	b.eq	1f3c <main+0x97c>  // b.none
    1f34:	f94633e0 	ldr	x0, [sp, #3168]
    1f38:	97fffcae 	bl	11f0 <fclose@plt>
		if(fPtr2 != NULL) fclose(fPtr2);
    1f3c:	f9462fe0 	ldr	x0, [sp, #3160]
    1f40:	f100001f 	cmp	x0, #0x0
    1f44:	54000060 	b.eq	1f50 <main+0x990>  // b.none
    1f48:	f9462fe0 	ldr	x0, [sp, #3160]
    1f4c:	97fffca9 	bl	11f0 <fclose@plt>

	}
	return 0;
    1f50:	52800000 	mov	w0, #0x0                   	// #0
}
    1f54:	f9400bf3 	ldr	x19, [sp, #16]
    1f58:	a9407bfd 	ldp	x29, x30, [sp]
    1f5c:	913383ff 	add	sp, sp, #0xce0
    1f60:	d65f03c0 	ret

0000000000001f64 <compareFile>:



int compareFile(FILE * fPtr1, FILE * fPtr2)
{
    1f64:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    1f68:	910003fd 	mov	x29, sp
    1f6c:	f9000fe0 	str	x0, [sp, #24]
    1f70:	f9000be1 	str	x1, [sp, #16]
	int ch1, ch2;
	do {
		ch1 = fgetc(fPtr1);
    1f74:	f9400fe0 	ldr	x0, [sp, #24]
    1f78:	97fffcb6 	bl	1250 <fgetc@plt>
    1f7c:	b9002fe0 	str	w0, [sp, #44]
		ch2 = fgetc(fPtr2);
    1f80:	f9400be0 	ldr	x0, [sp, #16]
    1f84:	97fffcb3 	bl	1250 <fgetc@plt>
    1f88:	b9002be0 	str	w0, [sp, #40]
		if (ch1 != ch2) {
    1f8c:	b9402fe1 	ldr	w1, [sp, #44]
    1f90:	b9402be0 	ldr	w0, [sp, #40]
    1f94:	6b00003f 	cmp	w1, w0
    1f98:	54000060 	b.eq	1fa4 <compareFile+0x40>  // b.none
			return -1;
    1f9c:	12800000 	mov	w0, #0xffffffff            	// #-1
    1fa0:	14000010 	b	1fe0 <compareFile+0x7c>
		}
	} while (ch1 != EOF && ch2 != EOF);
    1fa4:	b9402fe0 	ldr	w0, [sp, #44]
    1fa8:	3100041f 	cmn	w0, #0x1
    1fac:	54000080 	b.eq	1fbc <compareFile+0x58>  // b.none
    1fb0:	b9402be0 	ldr	w0, [sp, #40]
    1fb4:	3100041f 	cmn	w0, #0x1
    1fb8:	54fffde1 	b.ne	1f74 <compareFile+0x10>  // b.any

	/* If both files have reached end */
	if (ch1 == EOF && ch2 == EOF)
    1fbc:	b9402fe0 	ldr	w0, [sp, #44]
    1fc0:	3100041f 	cmn	w0, #0x1
    1fc4:	540000c1 	b.ne	1fdc <compareFile+0x78>  // b.any
    1fc8:	b9402be0 	ldr	w0, [sp, #40]
    1fcc:	3100041f 	cmn	w0, #0x1
    1fd0:	54000061 	b.ne	1fdc <compareFile+0x78>  // b.any
		return 0;
    1fd4:	52800000 	mov	w0, #0x0                   	// #0
    1fd8:	14000002 	b	1fe0 <compareFile+0x7c>
	else
		return -1;
    1fdc:	12800000 	mov	w0, #0xffffffff            	// #-1
}
    1fe0:	a8c37bfd 	ldp	x29, x30, [sp], #48
    1fe4:	d65f03c0 	ret

0000000000001fe8 <main._omp_fn.0>:
		#pragma omp parallel
    1fe8:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    1fec:	910003fd 	mov	x29, sp
    1ff0:	f9000fe0 	str	x0, [sp, #24]
            nthreads = omp_get_num_threads();
    1ff4:	97fffcaf 	bl	12b0 <omp_get_num_threads@plt>
    1ff8:	2a0003e1 	mov	w1, w0
    1ffc:	f9400fe0 	ldr	x0, [sp, #24]
    2000:	b9000001 	str	w1, [x0]
		#pragma omp parallel
    2004:	a8c27bfd 	ldp	x29, x30, [sp], #32
    2008:	d65f03c0 	ret

000000000000200c <sptNewSparseTensor>:
 * Create a new sparse tensor
 * @param tsr    a pointer to an uninitialized sparse tensor
 * @param nmodes number of modes the tensor will have
 * @param ndims  the dimension of each mode the tensor will have
 */
int sptNewSparseTensor(sptSparseTensor *tsr, sptIndex nmodes, const sptIndex ndims[]) {
    200c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2010:	910003fd 	mov	x29, sp
    2014:	f90017e0 	str	x0, [sp, #40]
    2018:	b90027e1 	str	w1, [sp, #36]
    201c:	f9000fe2 	str	x2, [sp, #24]
	sptIndex i;
	int result;
	tsr->nmodes = nmodes;
    2020:	f94017e0 	ldr	x0, [sp, #40]
    2024:	b94027e1 	ldr	w1, [sp, #36]
    2028:	b9000001 	str	w1, [x0]
	tsr->sortorder = malloc(nmodes * sizeof tsr->sortorder[0]);
    202c:	b94027e0 	ldr	w0, [sp, #36]
    2030:	d37ef400 	lsl	x0, x0, #2
    2034:	97fffc7b 	bl	1220 <malloc@plt>
    2038:	aa0003e1 	mov	x1, x0
    203c:	f94017e0 	ldr	x0, [sp, #40]
    2040:	f9000401 	str	x1, [x0, #8]
	for(i = 0; i < nmodes; ++i) {
    2044:	b9003fff 	str	wzr, [sp, #60]
    2048:	1400000b 	b	2074 <sptNewSparseTensor+0x68>
		tsr->sortorder[i] = i;
    204c:	f94017e0 	ldr	x0, [sp, #40]
    2050:	f9400401 	ldr	x1, [x0, #8]
    2054:	b9403fe0 	ldr	w0, [sp, #60]
    2058:	d37ef400 	lsl	x0, x0, #2
    205c:	8b000020 	add	x0, x1, x0
    2060:	b9403fe1 	ldr	w1, [sp, #60]
    2064:	b9000001 	str	w1, [x0]
	for(i = 0; i < nmodes; ++i) {
    2068:	b9403fe0 	ldr	w0, [sp, #60]
    206c:	11000400 	add	w0, w0, #0x1
    2070:	b9003fe0 	str	w0, [sp, #60]
    2074:	b9403fe1 	ldr	w1, [sp, #60]
    2078:	b94027e0 	ldr	w0, [sp, #36]
    207c:	6b00003f 	cmp	w1, w0
    2080:	54fffe63 	b.cc	204c <sptNewSparseTensor+0x40>  // b.lo, b.ul, b.last
	}
	tsr->ndims = malloc(nmodes * sizeof *tsr->ndims);
    2084:	b94027e0 	ldr	w0, [sp, #36]
    2088:	d37ef400 	lsl	x0, x0, #2
    208c:	97fffc65 	bl	1220 <malloc@plt>
    2090:	aa0003e1 	mov	x1, x0
    2094:	f94017e0 	ldr	x0, [sp, #40]
    2098:	f9000801 	str	x1, [x0, #16]
//	spt_CheckOSError(!tsr->ndims, "SpTns New");
	memcpy(tsr->ndims, ndims, nmodes * sizeof *tsr->ndims);
    209c:	f94017e0 	ldr	x0, [sp, #40]
    20a0:	f9400803 	ldr	x3, [x0, #16]
    20a4:	b94027e0 	ldr	w0, [sp, #36]
    20a8:	d37ef400 	lsl	x0, x0, #2
    20ac:	aa0003e2 	mov	x2, x0
    20b0:	f9400fe1 	ldr	x1, [sp, #24]
    20b4:	aa0303e0 	mov	x0, x3
    20b8:	97fffc2e 	bl	1170 <memcpy@plt>
	tsr->nnz = 0;
    20bc:	f94017e0 	ldr	x0, [sp, #40]
    20c0:	f9000c1f 	str	xzr, [x0, #24]
	tsr->inds = malloc(nmodes * sizeof *tsr->inds);
    20c4:	b94027e1 	ldr	w1, [sp, #36]
    20c8:	aa0103e0 	mov	x0, x1
    20cc:	d37ff800 	lsl	x0, x0, #1
    20d0:	8b010000 	add	x0, x0, x1
    20d4:	d37df000 	lsl	x0, x0, #3
    20d8:	97fffc52 	bl	1220 <malloc@plt>
    20dc:	aa0003e1 	mov	x1, x0
    20e0:	f94017e0 	ldr	x0, [sp, #40]
    20e4:	f9001001 	str	x1, [x0, #32]
//	spt_CheckOSError(!tsr->inds, "SpTns New");
	for(i = 0; i < nmodes; ++i) {
    20e8:	b9003fff 	str	wzr, [sp, #60]
    20ec:	14000021 	b	2170 <sptNewSparseTensor+0x164>
		result = sptNewIndexVector(&tsr->inds[i], 0, 0);
    20f0:	f94017e0 	ldr	x0, [sp, #40]
    20f4:	f9401002 	ldr	x2, [x0, #32]
    20f8:	b9403fe1 	ldr	w1, [sp, #60]
    20fc:	aa0103e0 	mov	x0, x1
    2100:	d37ff800 	lsl	x0, x0, #1
    2104:	8b010000 	add	x0, x0, x1
    2108:	d37df000 	lsl	x0, x0, #3
    210c:	8b000040 	add	x0, x2, x0
    2110:	d2800002 	mov	x2, #0x0                   	// #0
    2114:	d2800001 	mov	x1, #0x0                   	// #0
    2118:	94000166 	bl	26b0 <sptNewIndexVector>
    211c:	b9003be0 	str	w0, [sp, #56]
		spt_CheckError(result, "SpTns New", NULL);
    2120:	b9403be0 	ldr	w0, [sp, #56]
    2124:	7100001f 	cmp	w0, #0x0
    2128:	1a9f07e0 	cset	w0, ne  // ne = any
    212c:	12001c00 	and	w0, w0, #0xff
    2130:	92401c00 	and	x0, x0, #0xff
    2134:	f100001f 	cmp	x0, #0x0
    2138:	54000160 	b.eq	2164 <sptNewSparseTensor+0x158>  // b.none
    213c:	d2800004 	mov	x4, #0x0                   	// #0
    2140:	52800643 	mov	w3, #0x32                  	// #50
    2144:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2148:	911fa002 	add	x2, x0, #0x7e8
    214c:	b9403be1 	ldr	w1, [sp, #56]
    2150:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2154:	91204000 	add	x0, x0, #0x810
    2158:	94000f88 	bl	5f78 <spt_ComplainError>
    215c:	b9403be0 	ldr	w0, [sp, #56]
    2160:	14000020 	b	21e0 <sptNewSparseTensor+0x1d4>
	for(i = 0; i < nmodes; ++i) {
    2164:	b9403fe0 	ldr	w0, [sp, #60]
    2168:	11000400 	add	w0, w0, #0x1
    216c:	b9003fe0 	str	w0, [sp, #60]
    2170:	b9403fe1 	ldr	w1, [sp, #60]
    2174:	b94027e0 	ldr	w0, [sp, #36]
    2178:	6b00003f 	cmp	w1, w0
    217c:	54fffba3 	b.cc	20f0 <sptNewSparseTensor+0xe4>  // b.lo, b.ul, b.last
	}
	result = sptNewValueVector(&tsr->values, 0, 0);
    2180:	f94017e0 	ldr	x0, [sp, #40]
    2184:	9100a000 	add	x0, x0, #0x28
    2188:	d2800002 	mov	x2, #0x0                   	// #0
    218c:	d2800001 	mov	x1, #0x0                   	// #0
    2190:	9400003d 	bl	2284 <sptNewValueVector>
    2194:	b9003be0 	str	w0, [sp, #56]
	spt_CheckError(result, "SpTns New", NULL);
    2198:	b9403be0 	ldr	w0, [sp, #56]
    219c:	7100001f 	cmp	w0, #0x0
    21a0:	1a9f07e0 	cset	w0, ne  // ne = any
    21a4:	12001c00 	and	w0, w0, #0xff
    21a8:	92401c00 	and	x0, x0, #0xff
    21ac:	f100001f 	cmp	x0, #0x0
    21b0:	54000160 	b.eq	21dc <sptNewSparseTensor+0x1d0>  // b.none
    21b4:	d2800004 	mov	x4, #0x0                   	// #0
    21b8:	528006a3 	mov	w3, #0x35                  	// #53
    21bc:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    21c0:	911fa002 	add	x2, x0, #0x7e8
    21c4:	b9403be1 	ldr	w1, [sp, #56]
    21c8:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    21cc:	91204000 	add	x0, x0, #0x810
    21d0:	94000f6a 	bl	5f78 <spt_ComplainError>
    21d4:	b9403be0 	ldr	w0, [sp, #56]
    21d8:	14000002 	b	21e0 <sptNewSparseTensor+0x1d4>
	return 0;
    21dc:	52800000 	mov	w0, #0x0                   	// #0
}
    21e0:	a8c47bfd 	ldp	x29, x30, [sp], #64
    21e4:	d65f03c0 	ret

00000000000021e8 <sptFreeSparseTensor>:

/**
 * Release any memory the sparse tensor is holding
 * @param tsr the tensor to release
 */
void sptFreeSparseTensor(sptSparseTensor *tsr) {
    21e8:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    21ec:	910003fd 	mov	x29, sp
    21f0:	f9000fe0 	str	x0, [sp, #24]
	sptIndex i;
	for(i = 0; i < tsr->nmodes; ++i) {
    21f4:	b9002fff 	str	wzr, [sp, #44]
    21f8:	1400000d 	b	222c <sptFreeSparseTensor+0x44>
		sptFreeIndexVector(&tsr->inds[i]);
    21fc:	f9400fe0 	ldr	x0, [sp, #24]
    2200:	f9401002 	ldr	x2, [x0, #32]
    2204:	b9402fe1 	ldr	w1, [sp, #44]
    2208:	aa0103e0 	mov	x0, x1
    220c:	d37ff800 	lsl	x0, x0, #1
    2210:	8b010000 	add	x0, x0, x1
    2214:	d37df000 	lsl	x0, x0, #3
    2218:	8b000040 	add	x0, x2, x0
    221c:	9400020c 	bl	2a4c <sptFreeIndexVector>
	for(i = 0; i < tsr->nmodes; ++i) {
    2220:	b9402fe0 	ldr	w0, [sp, #44]
    2224:	11000400 	add	w0, w0, #0x1
    2228:	b9002fe0 	str	w0, [sp, #44]
    222c:	f9400fe0 	ldr	x0, [sp, #24]
    2230:	b9400000 	ldr	w0, [x0]
    2234:	b9402fe1 	ldr	w1, [sp, #44]
    2238:	6b00003f 	cmp	w1, w0
    223c:	54fffe03 	b.cc	21fc <sptFreeSparseTensor+0x14>  // b.lo, b.ul, b.last
	}
	free(tsr->sortorder);
    2240:	f9400fe0 	ldr	x0, [sp, #24]
    2244:	f9400400 	ldr	x0, [x0, #8]
    2248:	97fffc42 	bl	1350 <free@plt>
	free(tsr->ndims);
    224c:	f9400fe0 	ldr	x0, [sp, #24]
    2250:	f9400800 	ldr	x0, [x0, #16]
    2254:	97fffc3f 	bl	1350 <free@plt>
	free(tsr->inds);
    2258:	f9400fe0 	ldr	x0, [sp, #24]
    225c:	f9401000 	ldr	x0, [x0, #32]
    2260:	97fffc3c 	bl	1350 <free@plt>
	sptFreeValueVector(&tsr->values);
    2264:	f9400fe0 	ldr	x0, [sp, #24]
    2268:	9100a000 	add	x0, x0, #0x28
    226c:	94000104 	bl	267c <sptFreeValueVector>
	tsr->nmodes = 0;
    2270:	f9400fe0 	ldr	x0, [sp, #24]
    2274:	b900001f 	str	wzr, [x0]
}
    2278:	d503201f 	nop
    227c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2280:	d65f03c0 	ret

0000000000002284 <sptNewValueVector>:
 * @param len number of values to create
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptNewValueVector(sptValueVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    2284:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2288:	910003fd 	mov	x29, sp
    228c:	f9000bf3 	str	x19, [sp, #16]
    2290:	f9001fe0 	str	x0, [sp, #56]
    2294:	f9001be1 	str	x1, [sp, #48]
    2298:	f90017e2 	str	x2, [sp, #40]
	if(cap < len) {
    229c:	f94017e1 	ldr	x1, [sp, #40]
    22a0:	f9401be0 	ldr	x0, [sp, #48]
    22a4:	eb00003f 	cmp	x1, x0
    22a8:	54000062 	b.cs	22b4 <sptNewValueVector+0x30>  // b.hs, b.nlast
		cap = len;
    22ac:	f9401be0 	ldr	x0, [sp, #48]
    22b0:	f90017e0 	str	x0, [sp, #40]
	}
	if(cap < 2) {
    22b4:	f94017e0 	ldr	x0, [sp, #40]
    22b8:	f100041f 	cmp	x0, #0x1
    22bc:	54000068 	b.hi	22c8 <sptNewValueVector+0x44>  // b.pmore
		cap = 2;
    22c0:	d2800040 	mov	x0, #0x2                   	// #2
    22c4:	f90017e0 	str	x0, [sp, #40]
	}
	vec->len = len;
    22c8:	f9401fe0 	ldr	x0, [sp, #56]
    22cc:	f9401be1 	ldr	x1, [sp, #48]
    22d0:	f9000001 	str	x1, [x0]
	vec->cap = cap;
    22d4:	f9401fe0 	ldr	x0, [sp, #56]
    22d8:	f94017e1 	ldr	x1, [sp, #40]
    22dc:	f9000401 	str	x1, [x0, #8]
	vec->data = malloc(cap * sizeof *vec->data);
    22e0:	f94017e0 	ldr	x0, [sp, #40]
    22e4:	d37ef400 	lsl	x0, x0, #2
    22e8:	97fffbce 	bl	1220 <malloc@plt>
    22ec:	aa0003e1 	mov	x1, x0
    22f0:	f9401fe0 	ldr	x0, [sp, #56]
    22f4:	f9000801 	str	x1, [x0, #16]
	spt_CheckOSError(!vec->data, "ValVec New");
    22f8:	f9401fe0 	ldr	x0, [sp, #56]
    22fc:	f9400800 	ldr	x0, [x0, #16]
    2300:	f100001f 	cmp	x0, #0x0
    2304:	1a9f17e0 	cset	w0, eq  // eq = none
    2308:	12001c00 	and	w0, w0, #0xff
    230c:	92401c00 	and	x0, x0, #0xff
    2310:	f100001f 	cmp	x0, #0x0
    2314:	54000360 	b.eq	2380 <sptNewValueVector+0xfc>  // b.none
    2318:	97fffc2a 	bl	13c0 <__errno_location@plt>
    231c:	b9400000 	ldr	w0, [x0]
    2320:	3140401f 	cmn	w0, #0x10, lsl #12
    2324:	1a9f07e0 	cset	w0, ne  // ne = any
    2328:	12001c00 	and	w0, w0, #0xff
    232c:	92401c00 	and	x0, x0, #0xff
    2330:	f100001f 	cmp	x0, #0x0
    2334:	54000260 	b.eq	2380 <sptNewValueVector+0xfc>  // b.none
    2338:	97fffc22 	bl	13c0 <__errno_location@plt>
    233c:	b9400000 	ldr	w0, [x0]
    2340:	11404013 	add	w19, w0, #0x10, lsl #12
    2344:	97fffc1f 	bl	13c0 <__errno_location@plt>
    2348:	b9400000 	ldr	w0, [x0]
    234c:	97fffbdd 	bl	12c0 <strerror@plt>
    2350:	aa0003e4 	mov	x4, x0
    2354:	528005c3 	mov	w3, #0x2e                  	// #46
    2358:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    235c:	91208002 	add	x2, x0, #0x820
    2360:	2a1303e1 	mov	w1, w19
    2364:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2368:	91212000 	add	x0, x0, #0x848
    236c:	94000f03 	bl	5f78 <spt_ComplainError>
    2370:	97fffc14 	bl	13c0 <__errno_location@plt>
    2374:	b9400000 	ldr	w0, [x0]
    2378:	11404000 	add	w0, w0, #0x10, lsl #12
    237c:	1400000a 	b	23a4 <sptNewValueVector+0x120>
	memset(vec->data, 0, cap * sizeof *vec->data);
    2380:	f9401fe0 	ldr	x0, [sp, #56]
    2384:	f9400803 	ldr	x3, [x0, #16]
    2388:	f94017e0 	ldr	x0, [sp, #40]
    238c:	d37ef400 	lsl	x0, x0, #2
    2390:	aa0003e2 	mov	x2, x0
    2394:	52800001 	mov	w1, #0x0                   	// #0
    2398:	aa0303e0 	mov	x0, x3
    239c:	97fffbb5 	bl	1270 <memset@plt>
	return 0;
    23a0:	52800000 	mov	w0, #0x0                   	// #0
}
    23a4:	f9400bf3 	ldr	x19, [sp, #16]
    23a8:	a8c47bfd 	ldp	x29, x30, [sp], #64
    23ac:	d65f03c0 	ret

00000000000023b0 <sptConstantValueVector>:
 * @param vec   a valid pointer to an existed sptVector variable,
 * @param val   a given value constant
 *
 * Vector is a type of one-dimentional array with dynamic length
 */
int sptConstantValueVector(sptValueVector * const vec, sptValue const val) {
    23b0:	d10083ff 	sub	sp, sp, #0x20
    23b4:	f90007e0 	str	x0, [sp, #8]
    23b8:	bd0007e0 	str	s0, [sp, #4]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    23bc:	f9000fff 	str	xzr, [sp, #24]
    23c0:	1400000b 	b	23ec <sptConstantValueVector+0x3c>
		vec->data[i] = val;
    23c4:	f94007e0 	ldr	x0, [sp, #8]
    23c8:	f9400801 	ldr	x1, [x0, #16]
    23cc:	f9400fe0 	ldr	x0, [sp, #24]
    23d0:	d37ef400 	lsl	x0, x0, #2
    23d4:	8b000020 	add	x0, x1, x0
    23d8:	bd4007e0 	ldr	s0, [sp, #4]
    23dc:	bd000000 	str	s0, [x0]
	for(sptNnzIndex i=0; i<vec->len; ++i)
    23e0:	f9400fe0 	ldr	x0, [sp, #24]
    23e4:	91000400 	add	x0, x0, #0x1
    23e8:	f9000fe0 	str	x0, [sp, #24]
    23ec:	f94007e0 	ldr	x0, [sp, #8]
    23f0:	f9400000 	ldr	x0, [x0]
    23f4:	f9400fe1 	ldr	x1, [sp, #24]
    23f8:	eb00003f 	cmp	x1, x0
    23fc:	54fffe43 	b.cc	23c4 <sptConstantValueVector+0x14>  // b.lo, b.ul, b.last
	return 0;
    2400:	52800000 	mov	w0, #0x0                   	// #0
}
    2404:	910083ff 	add	sp, sp, #0x20
    2408:	d65f03c0 	ret

000000000000240c <sptAppendValueVector>:
 * @param vec   a pointer to a valid value vector
 * @param value the value to be appended
 *
 * The length of the value vector will be changed to contain the new value.
 */
int sptAppendValueVector(sptValueVector *vec, sptValue const value) {
    240c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2410:	910003fd 	mov	x29, sp
    2414:	f9000bf3 	str	x19, [sp, #16]
    2418:	f90017e0 	str	x0, [sp, #40]
    241c:	bd0027e0 	str	s0, [sp, #36]
	if(vec->cap <= vec->len) {
    2420:	f94017e0 	ldr	x0, [sp, #40]
    2424:	f9400401 	ldr	x1, [x0, #8]
    2428:	f94017e0 	ldr	x0, [sp, #40]
    242c:	f9400000 	ldr	x0, [x0]
    2430:	eb00003f 	cmp	x1, x0
    2434:	540006e8 	b.hi	2510 <sptAppendValueVector+0x104>  // b.pmore
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    2438:	f94017e0 	ldr	x0, [sp, #40]
    243c:	f9400401 	ldr	x1, [x0, #8]
    2440:	f94017e0 	ldr	x0, [sp, #40]
    2444:	f9400400 	ldr	x0, [x0, #8]
    2448:	d341fc00 	lsr	x0, x0, #1
    244c:	8b000020 	add	x0, x1, x0
    2450:	f9001fe0 	str	x0, [sp, #56]
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2454:	f94017e0 	ldr	x0, [sp, #40]
    2458:	f9400802 	ldr	x2, [x0, #16]
    245c:	f9401fe0 	ldr	x0, [sp, #56]
    2460:	d37ef400 	lsl	x0, x0, #2
    2464:	aa0003e1 	mov	x1, x0
    2468:	aa0203e0 	mov	x0, x2
    246c:	97fffb85 	bl	1280 <realloc@plt>
    2470:	f9001be0 	str	x0, [sp, #48]
		spt_CheckOSError(!newdata, "ValVec Append");
    2474:	f9401be0 	ldr	x0, [sp, #48]
    2478:	f100001f 	cmp	x0, #0x0
    247c:	1a9f17e0 	cset	w0, eq  // eq = none
    2480:	12001c00 	and	w0, w0, #0xff
    2484:	92401c00 	and	x0, x0, #0xff
    2488:	f100001f 	cmp	x0, #0x0
    248c:	54000360 	b.eq	24f8 <sptAppendValueVector+0xec>  // b.none
    2490:	97fffbcc 	bl	13c0 <__errno_location@plt>
    2494:	b9400000 	ldr	w0, [x0]
    2498:	3140401f 	cmn	w0, #0x10, lsl #12
    249c:	1a9f07e0 	cset	w0, ne  // ne = any
    24a0:	12001c00 	and	w0, w0, #0xff
    24a4:	92401c00 	and	x0, x0, #0xff
    24a8:	f100001f 	cmp	x0, #0x0
    24ac:	54000260 	b.eq	24f8 <sptAppendValueVector+0xec>  // b.none
    24b0:	97fffbc4 	bl	13c0 <__errno_location@plt>
    24b4:	b9400000 	ldr	w0, [x0]
    24b8:	11404013 	add	w19, w0, #0x10, lsl #12
    24bc:	97fffbc1 	bl	13c0 <__errno_location@plt>
    24c0:	b9400000 	ldr	w0, [x0]
    24c4:	97fffb7f 	bl	12c0 <strerror@plt>
    24c8:	aa0003e4 	mov	x4, x0
    24cc:	52800a43 	mov	w3, #0x52                  	// #82
    24d0:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    24d4:	91208002 	add	x2, x0, #0x820
    24d8:	2a1303e1 	mov	w1, w19
    24dc:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    24e0:	91216000 	add	x0, x0, #0x858
    24e4:	94000ea5 	bl	5f78 <spt_ComplainError>
    24e8:	97fffbb6 	bl	13c0 <__errno_location@plt>
    24ec:	b9400000 	ldr	w0, [x0]
    24f0:	11404000 	add	w0, w0, #0x10, lsl #12
    24f4:	14000015 	b	2548 <sptAppendValueVector+0x13c>
		vec->cap = newcap;
    24f8:	f94017e0 	ldr	x0, [sp, #40]
    24fc:	f9401fe1 	ldr	x1, [sp, #56]
    2500:	f9000401 	str	x1, [x0, #8]
		vec->data = newdata;
    2504:	f94017e0 	ldr	x0, [sp, #40]
    2508:	f9401be1 	ldr	x1, [sp, #48]
    250c:	f9000801 	str	x1, [x0, #16]
	}
	vec->data[vec->len] = value;
    2510:	f94017e0 	ldr	x0, [sp, #40]
    2514:	f9400801 	ldr	x1, [x0, #16]
    2518:	f94017e0 	ldr	x0, [sp, #40]
    251c:	f9400000 	ldr	x0, [x0]
    2520:	d37ef400 	lsl	x0, x0, #2
    2524:	8b000020 	add	x0, x1, x0
    2528:	bd4027e0 	ldr	s0, [sp, #36]
    252c:	bd000000 	str	s0, [x0]
	++vec->len;
    2530:	f94017e0 	ldr	x0, [sp, #40]
    2534:	f9400000 	ldr	x0, [x0]
    2538:	91000401 	add	x1, x0, #0x1
    253c:	f94017e0 	ldr	x0, [sp, #40]
    2540:	f9000001 	str	x1, [x0]
	return 0;
    2544:	52800000 	mov	w0, #0x0                   	// #0
}
    2548:	f9400bf3 	ldr	x19, [sp, #16]
    254c:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2550:	d65f03c0 	ret

0000000000002554 <sptResizeValueVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeValueVector(sptValueVector *vec, sptNnzIndex const size) {
    2554:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2558:	910003fd 	mov	x29, sp
    255c:	f9000bf3 	str	x19, [sp, #16]
    2560:	f90017e0 	str	x0, [sp, #40]
    2564:	f90013e1 	str	x1, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2568:	f94013e2 	ldr	x2, [sp, #32]
    256c:	f94013e1 	ldr	x1, [sp, #32]
    2570:	d2800040 	mov	x0, #0x2                   	// #2
    2574:	f100085f 	cmp	x2, #0x2
    2578:	9a802020 	csel	x0, x1, x0, cs  // cs = hs, nlast
    257c:	f9001fe0 	str	x0, [sp, #56]
	if(newcap != vec->cap) {
    2580:	f94017e0 	ldr	x0, [sp, #40]
    2584:	f9400400 	ldr	x0, [x0, #8]
    2588:	f9401fe1 	ldr	x1, [sp, #56]
    258c:	eb00003f 	cmp	x1, x0
    2590:	54000680 	b.eq	2660 <sptResizeValueVector+0x10c>  // b.none
		sptValue *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2594:	f94017e0 	ldr	x0, [sp, #40]
    2598:	f9400802 	ldr	x2, [x0, #16]
    259c:	f9401fe0 	ldr	x0, [sp, #56]
    25a0:	d37ef400 	lsl	x0, x0, #2
    25a4:	aa0003e1 	mov	x1, x0
    25a8:	aa0203e0 	mov	x0, x2
    25ac:	97fffb35 	bl	1280 <realloc@plt>
    25b0:	f9001be0 	str	x0, [sp, #48]
		spt_CheckOSError(!newdata, "ValVec Resize");
    25b4:	f9401be0 	ldr	x0, [sp, #48]
    25b8:	f100001f 	cmp	x0, #0x0
    25bc:	1a9f17e0 	cset	w0, eq  // eq = none
    25c0:	12001c00 	and	w0, w0, #0xff
    25c4:	92401c00 	and	x0, x0, #0xff
    25c8:	f100001f 	cmp	x0, #0x0
    25cc:	54000360 	b.eq	2638 <sptResizeValueVector+0xe4>  // b.none
    25d0:	97fffb7c 	bl	13c0 <__errno_location@plt>
    25d4:	b9400000 	ldr	w0, [x0]
    25d8:	3140401f 	cmn	w0, #0x10, lsl #12
    25dc:	1a9f07e0 	cset	w0, ne  // ne = any
    25e0:	12001c00 	and	w0, w0, #0xff
    25e4:	92401c00 	and	x0, x0, #0xff
    25e8:	f100001f 	cmp	x0, #0x0
    25ec:	54000260 	b.eq	2638 <sptResizeValueVector+0xe4>  // b.none
    25f0:	97fffb74 	bl	13c0 <__errno_location@plt>
    25f4:	b9400000 	ldr	w0, [x0]
    25f8:	11404013 	add	w19, w0, #0x10, lsl #12
    25fc:	97fffb71 	bl	13c0 <__errno_location@plt>
    2600:	b9400000 	ldr	w0, [x0]
    2604:	97fffb2f 	bl	12c0 <strerror@plt>
    2608:	aa0003e4 	mov	x4, x0
    260c:	52800d23 	mov	w3, #0x69                  	// #105
    2610:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2614:	91208002 	add	x2, x0, #0x820
    2618:	2a1303e1 	mov	w1, w19
    261c:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2620:	9121a000 	add	x0, x0, #0x868
    2624:	94000e55 	bl	5f78 <spt_ComplainError>
    2628:	97fffb66 	bl	13c0 <__errno_location@plt>
    262c:	b9400000 	ldr	w0, [x0]
    2630:	11404000 	add	w0, w0, #0x10, lsl #12
    2634:	1400000f 	b	2670 <sptResizeValueVector+0x11c>
		vec->len = size;
    2638:	f94017e0 	ldr	x0, [sp, #40]
    263c:	f94013e1 	ldr	x1, [sp, #32]
    2640:	f9000001 	str	x1, [x0]
		vec->cap = newcap;
    2644:	f94017e0 	ldr	x0, [sp, #40]
    2648:	f9401fe1 	ldr	x1, [sp, #56]
    264c:	f9000401 	str	x1, [x0, #8]
		vec->data = newdata;
    2650:	f94017e0 	ldr	x0, [sp, #40]
    2654:	f9401be1 	ldr	x1, [sp, #48]
    2658:	f9000801 	str	x1, [x0, #16]
    265c:	14000004 	b	266c <sptResizeValueVector+0x118>
	} else {
		vec->len = size;
    2660:	f94017e0 	ldr	x0, [sp, #40]
    2664:	f94013e1 	ldr	x1, [sp, #32]
    2668:	f9000001 	str	x1, [x0]
	}
	return 0;
    266c:	52800000 	mov	w0, #0x0                   	// #0
}
    2670:	f9400bf3 	ldr	x19, [sp, #16]
    2674:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2678:	d65f03c0 	ret

000000000000267c <sptFreeValueVector>:
 * Release the memory buffer a value vector is holding
 *
 * @param vec a pointer to a valid value vector
 *
 */
void sptFreeValueVector(sptValueVector *vec) {
    267c:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2680:	910003fd 	mov	x29, sp
    2684:	f9000fe0 	str	x0, [sp, #24]
	vec->len = 0;
    2688:	f9400fe0 	ldr	x0, [sp, #24]
    268c:	f900001f 	str	xzr, [x0]
	vec->cap = 0;
    2690:	f9400fe0 	ldr	x0, [sp, #24]
    2694:	f900041f 	str	xzr, [x0, #8]
	free(vec->data);
    2698:	f9400fe0 	ldr	x0, [sp, #24]
    269c:	f9400800 	ldr	x0, [x0, #16]
    26a0:	97fffb2c 	bl	1350 <free@plt>
}
    26a4:	d503201f 	nop
    26a8:	a8c27bfd 	ldp	x29, x30, [sp], #32
    26ac:	d65f03c0 	ret

00000000000026b0 <sptNewIndexVector>:
 * @param cap total number of values to reserve
 *
 * Vector is a type of one-dimentional array with dynamic length
 */

int sptNewIndexVector(sptIndexVector *vec, sptNnzIndex len, sptNnzIndex cap) {
    26b0:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    26b4:	910003fd 	mov	x29, sp
    26b8:	f9000bf3 	str	x19, [sp, #16]
    26bc:	f9001fe0 	str	x0, [sp, #56]
    26c0:	f9001be1 	str	x1, [sp, #48]
    26c4:	f90017e2 	str	x2, [sp, #40]
	if(cap < len) {
    26c8:	f94017e1 	ldr	x1, [sp, #40]
    26cc:	f9401be0 	ldr	x0, [sp, #48]
    26d0:	eb00003f 	cmp	x1, x0
    26d4:	54000062 	b.cs	26e0 <sptNewIndexVector+0x30>  // b.hs, b.nlast
		cap = len;
    26d8:	f9401be0 	ldr	x0, [sp, #48]
    26dc:	f90017e0 	str	x0, [sp, #40]
	}
	if(cap < 2) {
    26e0:	f94017e0 	ldr	x0, [sp, #40]
    26e4:	f100041f 	cmp	x0, #0x1
    26e8:	54000068 	b.hi	26f4 <sptNewIndexVector+0x44>  // b.pmore
		cap = 2;
    26ec:	d2800040 	mov	x0, #0x2                   	// #2
    26f0:	f90017e0 	str	x0, [sp, #40]
	}
	vec->len = len;
    26f4:	f9401fe0 	ldr	x0, [sp, #56]
    26f8:	f9401be1 	ldr	x1, [sp, #48]
    26fc:	f9000001 	str	x1, [x0]
	vec->cap = cap;
    2700:	f9401fe0 	ldr	x0, [sp, #56]
    2704:	f94017e1 	ldr	x1, [sp, #40]
    2708:	f9000401 	str	x1, [x0, #8]
	vec->data = malloc(cap * sizeof *vec->data);
    270c:	f94017e0 	ldr	x0, [sp, #40]
    2710:	d37ef400 	lsl	x0, x0, #2
    2714:	97fffac3 	bl	1220 <malloc@plt>
    2718:	aa0003e1 	mov	x1, x0
    271c:	f9401fe0 	ldr	x0, [sp, #56]
    2720:	f9000801 	str	x1, [x0, #16]
	spt_CheckOSError(!vec->data, "IdxVec New");
    2724:	f9401fe0 	ldr	x0, [sp, #56]
    2728:	f9400800 	ldr	x0, [x0, #16]
    272c:	f100001f 	cmp	x0, #0x0
    2730:	1a9f17e0 	cset	w0, eq  // eq = none
    2734:	12001c00 	and	w0, w0, #0xff
    2738:	92401c00 	and	x0, x0, #0xff
    273c:	f100001f 	cmp	x0, #0x0
    2740:	54000360 	b.eq	27ac <sptNewIndexVector+0xfc>  // b.none
    2744:	97fffb1f 	bl	13c0 <__errno_location@plt>
    2748:	b9400000 	ldr	w0, [x0]
    274c:	3140401f 	cmn	w0, #0x10, lsl #12
    2750:	1a9f07e0 	cset	w0, ne  // ne = any
    2754:	12001c00 	and	w0, w0, #0xff
    2758:	92401c00 	and	x0, x0, #0xff
    275c:	f100001f 	cmp	x0, #0x0
    2760:	54000260 	b.eq	27ac <sptNewIndexVector+0xfc>  // b.none
    2764:	97fffb17 	bl	13c0 <__errno_location@plt>
    2768:	b9400000 	ldr	w0, [x0]
    276c:	11404013 	add	w19, w0, #0x10, lsl #12
    2770:	97fffb14 	bl	13c0 <__errno_location@plt>
    2774:	b9400000 	ldr	w0, [x0]
    2778:	97fffad2 	bl	12c0 <strerror@plt>
    277c:	aa0003e4 	mov	x4, x0
    2780:	52801283 	mov	w3, #0x94                  	// #148
    2784:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2788:	91208002 	add	x2, x0, #0x820
    278c:	2a1303e1 	mov	w1, w19
    2790:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2794:	9121e000 	add	x0, x0, #0x878
    2798:	94000df8 	bl	5f78 <spt_ComplainError>
    279c:	97fffb09 	bl	13c0 <__errno_location@plt>
    27a0:	b9400000 	ldr	w0, [x0]
    27a4:	11404000 	add	w0, w0, #0x10, lsl #12
    27a8:	1400000a 	b	27d0 <sptNewIndexVector+0x120>
	memset(vec->data, 0, cap * sizeof *vec->data);
    27ac:	f9401fe0 	ldr	x0, [sp, #56]
    27b0:	f9400803 	ldr	x3, [x0, #16]
    27b4:	f94017e0 	ldr	x0, [sp, #40]
    27b8:	d37ef400 	lsl	x0, x0, #2
    27bc:	aa0003e2 	mov	x2, x0
    27c0:	52800001 	mov	w1, #0x0                   	// #0
    27c4:	aa0303e0 	mov	x0, x3
    27c8:	97fffaaa 	bl	1270 <memset@plt>
	return 0;
    27cc:	52800000 	mov	w0, #0x0                   	// #0
}
    27d0:	f9400bf3 	ldr	x19, [sp, #16]
    27d4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    27d8:	d65f03c0 	ret

00000000000027dc <sptAppendIndexVector>:
 * @param vec   a pointer to a valid index vector
 * @param value the value to be appended
 *
 * The length of the size vector will be changed to contain the new value.
 */
int sptAppendIndexVector(sptIndexVector *vec, sptIndex const value) {
    27dc:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    27e0:	910003fd 	mov	x29, sp
    27e4:	f9000bf3 	str	x19, [sp, #16]
    27e8:	f90017e0 	str	x0, [sp, #40]
    27ec:	b90027e1 	str	w1, [sp, #36]
	if(vec->cap <= vec->len) {
    27f0:	f94017e0 	ldr	x0, [sp, #40]
    27f4:	f9400401 	ldr	x1, [x0, #8]
    27f8:	f94017e0 	ldr	x0, [sp, #40]
    27fc:	f9400000 	ldr	x0, [x0]
    2800:	eb00003f 	cmp	x1, x0
    2804:	540006e8 	b.hi	28e0 <sptAppendIndexVector+0x104>  // b.pmore
#ifndef MEMCHECK_MODE
		sptNnzIndex newcap = vec->cap + vec->cap/2;
    2808:	f94017e0 	ldr	x0, [sp, #40]
    280c:	f9400401 	ldr	x1, [x0, #8]
    2810:	f94017e0 	ldr	x0, [sp, #40]
    2814:	f9400400 	ldr	x0, [x0, #8]
    2818:	d341fc00 	lsr	x0, x0, #1
    281c:	8b000020 	add	x0, x1, x0
    2820:	f9001fe0 	str	x0, [sp, #56]
#else
		sptNnzIndex newcap = vec->len+1;
#endif
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2824:	f94017e0 	ldr	x0, [sp, #40]
    2828:	f9400802 	ldr	x2, [x0, #16]
    282c:	f9401fe0 	ldr	x0, [sp, #56]
    2830:	d37ef400 	lsl	x0, x0, #2
    2834:	aa0003e1 	mov	x1, x0
    2838:	aa0203e0 	mov	x0, x2
    283c:	97fffa91 	bl	1280 <realloc@plt>
    2840:	f9001be0 	str	x0, [sp, #48]
		spt_CheckOSError(!newdata, "IdxVec Append");
    2844:	f9401be0 	ldr	x0, [sp, #48]
    2848:	f100001f 	cmp	x0, #0x0
    284c:	1a9f17e0 	cset	w0, eq  // eq = none
    2850:	12001c00 	and	w0, w0, #0xff
    2854:	92401c00 	and	x0, x0, #0xff
    2858:	f100001f 	cmp	x0, #0x0
    285c:	54000360 	b.eq	28c8 <sptAppendIndexVector+0xec>  // b.none
    2860:	97fffad8 	bl	13c0 <__errno_location@plt>
    2864:	b9400000 	ldr	w0, [x0]
    2868:	3140401f 	cmn	w0, #0x10, lsl #12
    286c:	1a9f07e0 	cset	w0, ne  // ne = any
    2870:	12001c00 	and	w0, w0, #0xff
    2874:	92401c00 	and	x0, x0, #0xff
    2878:	f100001f 	cmp	x0, #0x0
    287c:	54000260 	b.eq	28c8 <sptAppendIndexVector+0xec>  // b.none
    2880:	97fffad0 	bl	13c0 <__errno_location@plt>
    2884:	b9400000 	ldr	w0, [x0]
    2888:	11404013 	add	w19, w0, #0x10, lsl #12
    288c:	97fffacd 	bl	13c0 <__errno_location@plt>
    2890:	b9400000 	ldr	w0, [x0]
    2894:	97fffa8b 	bl	12c0 <strerror@plt>
    2898:	aa0003e4 	mov	x4, x0
    289c:	52801543 	mov	w3, #0xaa                  	// #170
    28a0:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    28a4:	91208002 	add	x2, x0, #0x820
    28a8:	2a1303e1 	mov	w1, w19
    28ac:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    28b0:	91222000 	add	x0, x0, #0x888
    28b4:	94000db1 	bl	5f78 <spt_ComplainError>
    28b8:	97fffac2 	bl	13c0 <__errno_location@plt>
    28bc:	b9400000 	ldr	w0, [x0]
    28c0:	11404000 	add	w0, w0, #0x10, lsl #12
    28c4:	14000015 	b	2918 <sptAppendIndexVector+0x13c>
		vec->cap = newcap;
    28c8:	f94017e0 	ldr	x0, [sp, #40]
    28cc:	f9401fe1 	ldr	x1, [sp, #56]
    28d0:	f9000401 	str	x1, [x0, #8]
		vec->data = newdata;
    28d4:	f94017e0 	ldr	x0, [sp, #40]
    28d8:	f9401be1 	ldr	x1, [sp, #48]
    28dc:	f9000801 	str	x1, [x0, #16]
	}
	vec->data[vec->len] = value;
    28e0:	f94017e0 	ldr	x0, [sp, #40]
    28e4:	f9400801 	ldr	x1, [x0, #16]
    28e8:	f94017e0 	ldr	x0, [sp, #40]
    28ec:	f9400000 	ldr	x0, [x0]
    28f0:	d37ef400 	lsl	x0, x0, #2
    28f4:	8b000020 	add	x0, x1, x0
    28f8:	b94027e1 	ldr	w1, [sp, #36]
    28fc:	b9000001 	str	w1, [x0]
	++vec->len;
    2900:	f94017e0 	ldr	x0, [sp, #40]
    2904:	f9400000 	ldr	x0, [x0]
    2908:	91000401 	add	x1, x0, #0x1
    290c:	f94017e0 	ldr	x0, [sp, #40]
    2910:	f9000001 	str	x1, [x0]
	return 0;
    2914:	52800000 	mov	w0, #0x0                   	// #0
}
    2918:	f9400bf3 	ldr	x19, [sp, #16]
    291c:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2920:	d65f03c0 	ret

0000000000002924 <sptResizeIndexVector>:
 *
 * If the new size is larger than the current size, new values will be appended
 * but the values of them are undefined. If the new size if smaller than the
 * current size, values at the end will be truncated.
 */
int sptResizeIndexVector(sptIndexVector *vec, sptNnzIndex const size) {
    2924:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    2928:	910003fd 	mov	x29, sp
    292c:	f9000bf3 	str	x19, [sp, #16]
    2930:	f90017e0 	str	x0, [sp, #40]
    2934:	f90013e1 	str	x1, [sp, #32]
	sptNnzIndex newcap = size < 2 ? 2 : size;
    2938:	f94013e2 	ldr	x2, [sp, #32]
    293c:	f94013e1 	ldr	x1, [sp, #32]
    2940:	d2800040 	mov	x0, #0x2                   	// #2
    2944:	f100085f 	cmp	x2, #0x2
    2948:	9a802020 	csel	x0, x1, x0, cs  // cs = hs, nlast
    294c:	f9001fe0 	str	x0, [sp, #56]
	if(newcap != vec->cap) {
    2950:	f94017e0 	ldr	x0, [sp, #40]
    2954:	f9400400 	ldr	x0, [x0, #8]
    2958:	f9401fe1 	ldr	x1, [sp, #56]
    295c:	eb00003f 	cmp	x1, x0
    2960:	54000680 	b.eq	2a30 <sptResizeIndexVector+0x10c>  // b.none
		sptIndex *newdata = realloc(vec->data, newcap * sizeof *vec->data);
    2964:	f94017e0 	ldr	x0, [sp, #40]
    2968:	f9400802 	ldr	x2, [x0, #16]
    296c:	f9401fe0 	ldr	x0, [sp, #56]
    2970:	d37ef400 	lsl	x0, x0, #2
    2974:	aa0003e1 	mov	x1, x0
    2978:	aa0203e0 	mov	x0, x2
    297c:	97fffa41 	bl	1280 <realloc@plt>
    2980:	f9001be0 	str	x0, [sp, #48]
		spt_CheckOSError(!newdata, "IdxVec Resize");
    2984:	f9401be0 	ldr	x0, [sp, #48]
    2988:	f100001f 	cmp	x0, #0x0
    298c:	1a9f17e0 	cset	w0, eq  // eq = none
    2990:	12001c00 	and	w0, w0, #0xff
    2994:	92401c00 	and	x0, x0, #0xff
    2998:	f100001f 	cmp	x0, #0x0
    299c:	54000360 	b.eq	2a08 <sptResizeIndexVector+0xe4>  // b.none
    29a0:	97fffa88 	bl	13c0 <__errno_location@plt>
    29a4:	b9400000 	ldr	w0, [x0]
    29a8:	3140401f 	cmn	w0, #0x10, lsl #12
    29ac:	1a9f07e0 	cset	w0, ne  // ne = any
    29b0:	12001c00 	and	w0, w0, #0xff
    29b4:	92401c00 	and	x0, x0, #0xff
    29b8:	f100001f 	cmp	x0, #0x0
    29bc:	54000260 	b.eq	2a08 <sptResizeIndexVector+0xe4>  // b.none
    29c0:	97fffa80 	bl	13c0 <__errno_location@plt>
    29c4:	b9400000 	ldr	w0, [x0]
    29c8:	11404013 	add	w19, w0, #0x10, lsl #12
    29cc:	97fffa7d 	bl	13c0 <__errno_location@plt>
    29d0:	b9400000 	ldr	w0, [x0]
    29d4:	97fffa3b 	bl	12c0 <strerror@plt>
    29d8:	aa0003e4 	mov	x4, x0
    29dc:	52801823 	mov	w3, #0xc1                  	// #193
    29e0:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    29e4:	91208002 	add	x2, x0, #0x820
    29e8:	2a1303e1 	mov	w1, w19
    29ec:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    29f0:	91226000 	add	x0, x0, #0x898
    29f4:	94000d61 	bl	5f78 <spt_ComplainError>
    29f8:	97fffa72 	bl	13c0 <__errno_location@plt>
    29fc:	b9400000 	ldr	w0, [x0]
    2a00:	11404000 	add	w0, w0, #0x10, lsl #12
    2a04:	1400000f 	b	2a40 <sptResizeIndexVector+0x11c>
		vec->len = size;
    2a08:	f94017e0 	ldr	x0, [sp, #40]
    2a0c:	f94013e1 	ldr	x1, [sp, #32]
    2a10:	f9000001 	str	x1, [x0]
		vec->cap = newcap;
    2a14:	f94017e0 	ldr	x0, [sp, #40]
    2a18:	f9401fe1 	ldr	x1, [sp, #56]
    2a1c:	f9000401 	str	x1, [x0, #8]
		vec->data = newdata;
    2a20:	f94017e0 	ldr	x0, [sp, #40]
    2a24:	f9401be1 	ldr	x1, [sp, #48]
    2a28:	f9000801 	str	x1, [x0, #16]
    2a2c:	14000004 	b	2a3c <sptResizeIndexVector+0x118>
	} else {
		vec->len = size;
    2a30:	f94017e0 	ldr	x0, [sp, #40]
    2a34:	f94013e1 	ldr	x1, [sp, #32]
    2a38:	f9000001 	str	x1, [x0]
	}
	return 0;
    2a3c:	52800000 	mov	w0, #0x0                   	// #0
}
    2a40:	f9400bf3 	ldr	x19, [sp, #16]
    2a44:	a8c47bfd 	ldp	x29, x30, [sp], #64
    2a48:	d65f03c0 	ret

0000000000002a4c <sptFreeIndexVector>:
 * Release the memory buffer a sptIndexVector is holding
 *
 * @param vec a pointer to a valid size vector
 *
 */
void sptFreeIndexVector(sptIndexVector *vec) {
    2a4c:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    2a50:	910003fd 	mov	x29, sp
    2a54:	f9000fe0 	str	x0, [sp, #24]
	free(vec->data);
    2a58:	f9400fe0 	ldr	x0, [sp, #24]
    2a5c:	f9400800 	ldr	x0, [x0, #16]
    2a60:	97fffa3c 	bl	1350 <free@plt>
	vec->len = 0;
    2a64:	f9400fe0 	ldr	x0, [sp, #24]
    2a68:	f900001f 	str	xzr, [x0]
	vec->cap = 0;
    2a6c:	f9400fe0 	ldr	x0, [sp, #24]
    2a70:	f900041f 	str	xzr, [x0, #8]
}
    2a74:	d503201f 	nop
    2a78:	a8c27bfd 	ldp	x29, x30, [sp], #32
    2a7c:	d65f03c0 	ret

0000000000002a80 <get_file_type>:
};


static int get_file_type(
		char const * const fname)
{
    2a80:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    2a84:	910003fd 	mov	x29, sp
    2a88:	f9000fe0 	str	x0, [sp, #24]
	/* find last . in filename */
	char const * const suffix = strrchr(fname, '.');
    2a8c:	528005c1 	mov	w1, #0x2e                  	// #46
    2a90:	f9400fe0 	ldr	x0, [sp, #24]
    2a94:	97fffa0f 	bl	12d0 <strrchr@plt>
    2a98:	f90013e0 	str	x0, [sp, #32]
	if(suffix == NULL) {
    2a9c:	f94013e0 	ldr	x0, [sp, #32]
    2aa0:	f100001f 	cmp	x0, #0x0
    2aa4:	54000400 	b.eq	2b24 <get_file_type+0xa4>  // b.none
		goto NOT_FOUND;
	}

	size_t idx = 0;
    2aa8:	f90017ff 	str	xzr, [sp, #40]
	do {
		if(strcmp(suffix, file_extensions[idx].extension) == 0) {
    2aac:	d00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    2ab0:	9109e001 	add	x1, x0, #0x278
    2ab4:	f94017e0 	ldr	x0, [sp, #40]
    2ab8:	d37cec00 	lsl	x0, x0, #4
    2abc:	8b000020 	add	x0, x1, x0
    2ac0:	f9400000 	ldr	x0, [x0]
    2ac4:	aa0003e1 	mov	x1, x0
    2ac8:	f94013e0 	ldr	x0, [sp, #32]
    2acc:	97fffa19 	bl	1330 <strcmp@plt>
    2ad0:	7100001f 	cmp	w0, #0x0
    2ad4:	54000101 	b.ne	2af4 <get_file_type+0x74>  // b.any
			return file_extensions[idx].type;
    2ad8:	d00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    2adc:	9109e001 	add	x1, x0, #0x278
    2ae0:	f94017e0 	ldr	x0, [sp, #40]
    2ae4:	d37cec00 	lsl	x0, x0, #4
    2ae8:	8b000020 	add	x0, x1, x0
    2aec:	b9400800 	ldr	w0, [x0, #8]
    2af0:	14000017 	b	2b4c <get_file_type+0xcc>
		}
	} while(file_extensions[++idx].extension != NULL);
    2af4:	f94017e0 	ldr	x0, [sp, #40]
    2af8:	91000400 	add	x0, x0, #0x1
    2afc:	f90017e0 	str	x0, [sp, #40]
    2b00:	d00000a0 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    2b04:	9109e001 	add	x1, x0, #0x278
    2b08:	f94017e0 	ldr	x0, [sp, #40]
    2b0c:	d37cec00 	lsl	x0, x0, #4
    2b10:	8b000020 	add	x0, x1, x0
    2b14:	f9400000 	ldr	x0, [x0]
    2b18:	f100001f 	cmp	x0, #0x0
    2b1c:	54fffc81 	b.ne	2aac <get_file_type+0x2c>  // b.any


	/* default to text coordinate format */
	NOT_FOUND:
    2b20:	14000002 	b	2b28 <get_file_type+0xa8>
		goto NOT_FOUND;
    2b24:	d503201f 	nop
	fprintf(stderr, "SPLATT: extension for '%s' not recognized. "
    2b28:	b00000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    2b2c:	f947d800 	ldr	x0, [x0, #4016]
    2b30:	f9400003 	ldr	x3, [x0]
    2b34:	f9400fe2 	ldr	x2, [sp, #24]
    2b38:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2b3c:	91230001 	add	x1, x0, #0x8c0
    2b40:	aa0303e0 	mov	x0, x3
    2b44:	97fffa2b 	bl	13f0 <fprintf@plt>
									"Defaulting to ASCII coordinate form.\n", fname);
	return 0;
    2b48:	52800000 	mov	w0, #0x0                   	// #0
}
    2b4c:	a8c37bfd 	ldp	x29, x30, [sp], #48
    2b50:	d65f03c0 	ret

0000000000002b54 <p_tt_read_file>:

static int p_tt_read_file(sptSparseTensor *tsr, sptIndex start_index, FILE *fp)
{
    2b54:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    2b58:	910003fd 	mov	x29, sp
    2b5c:	f9000bf3 	str	x19, [sp, #16]
    2b60:	f9001fe0 	str	x0, [sp, #56]
    2b64:	b90037e1 	str	w1, [sp, #52]
    2b68:	f90017e2 	str	x2, [sp, #40]
	int iores, retval;
	sptIndex mode;
	iores = fscanf(fp, "%u", &tsr->nmodes);
    2b6c:	f9401fe0 	ldr	x0, [sp, #56]
    2b70:	aa0003e2 	mov	x2, x0
    2b74:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2b78:	91246001 	add	x1, x0, #0x918
    2b7c:	f94017e0 	ldr	x0, [sp, #40]
    2b80:	97fff9ac 	bl	1230 <__isoc99_fscanf@plt>
    2b84:	b90057e0 	str	w0, [sp, #84]
	spt_CheckOSError(iores < 0, "SpTns Load");
    2b88:	b94057e0 	ldr	w0, [sp, #84]
    2b8c:	531f7c00 	lsr	w0, w0, #31
    2b90:	12001c00 	and	w0, w0, #0xff
    2b94:	92401c00 	and	x0, x0, #0xff
    2b98:	f100001f 	cmp	x0, #0x0
    2b9c:	54000360 	b.eq	2c08 <p_tt_read_file+0xb4>  // b.none
    2ba0:	97fffa08 	bl	13c0 <__errno_location@plt>
    2ba4:	b9400000 	ldr	w0, [x0]
    2ba8:	3140401f 	cmn	w0, #0x10, lsl #12
    2bac:	1a9f07e0 	cset	w0, ne  // ne = any
    2bb0:	12001c00 	and	w0, w0, #0xff
    2bb4:	92401c00 	and	x0, x0, #0xff
    2bb8:	f100001f 	cmp	x0, #0x0
    2bbc:	54000260 	b.eq	2c08 <p_tt_read_file+0xb4>  // b.none
    2bc0:	97fffa00 	bl	13c0 <__errno_location@plt>
    2bc4:	b9400000 	ldr	w0, [x0]
    2bc8:	11404013 	add	w19, w0, #0x10, lsl #12
    2bcc:	97fff9fd 	bl	13c0 <__errno_location@plt>
    2bd0:	b9400000 	ldr	w0, [x0]
    2bd4:	97fff9bb 	bl	12c0 <strerror@plt>
    2bd8:	aa0003e4 	mov	x4, x0
    2bdc:	52800943 	mov	w3, #0x4a                  	// #74
    2be0:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2be4:	91248002 	add	x2, x0, #0x920
    2be8:	2a1303e1 	mov	w1, w19
    2bec:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2bf0:	91252000 	add	x0, x0, #0x948
    2bf4:	94000ce1 	bl	5f78 <spt_ComplainError>
    2bf8:	97fff9f2 	bl	13c0 <__errno_location@plt>
    2bfc:	b9400000 	ldr	w0, [x0]
    2c00:	11404000 	add	w0, w0, #0x10, lsl #12
    2c04:	14000169 	b	31a8 <p_tt_read_file+0x654>
	/* Only allocate space for sortorder, initialized to 0s. */
	tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    2c08:	f9401fe0 	ldr	x0, [sp, #56]
    2c0c:	b9400000 	ldr	w0, [x0]
    2c10:	2a0003e0 	mov	w0, w0
    2c14:	d37ef400 	lsl	x0, x0, #2
    2c18:	97fff982 	bl	1220 <malloc@plt>
    2c1c:	aa0003e1 	mov	x1, x0
    2c20:	f9401fe0 	ldr	x0, [sp, #56]
    2c24:	f9000401 	str	x1, [x0, #8]
	spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    2c28:	f9401fe0 	ldr	x0, [sp, #56]
    2c2c:	f9400400 	ldr	x0, [x0, #8]
    2c30:	f100001f 	cmp	x0, #0x0
    2c34:	1a9f17e0 	cset	w0, eq  // eq = none
    2c38:	12001c00 	and	w0, w0, #0xff
    2c3c:	92401c00 	and	x0, x0, #0xff
    2c40:	f100001f 	cmp	x0, #0x0
    2c44:	54000360 	b.eq	2cb0 <p_tt_read_file+0x15c>  // b.none
    2c48:	97fff9de 	bl	13c0 <__errno_location@plt>
    2c4c:	b9400000 	ldr	w0, [x0]
    2c50:	3140401f 	cmn	w0, #0x10, lsl #12
    2c54:	1a9f07e0 	cset	w0, ne  // ne = any
    2c58:	12001c00 	and	w0, w0, #0xff
    2c5c:	92401c00 	and	x0, x0, #0xff
    2c60:	f100001f 	cmp	x0, #0x0
    2c64:	54000260 	b.eq	2cb0 <p_tt_read_file+0x15c>  // b.none
    2c68:	97fff9d6 	bl	13c0 <__errno_location@plt>
    2c6c:	b9400000 	ldr	w0, [x0]
    2c70:	11404013 	add	w19, w0, #0x10, lsl #12
    2c74:	97fff9d3 	bl	13c0 <__errno_location@plt>
    2c78:	b9400000 	ldr	w0, [x0]
    2c7c:	97fff991 	bl	12c0 <strerror@plt>
    2c80:	aa0003e4 	mov	x4, x0
    2c84:	528009a3 	mov	w3, #0x4d                  	// #77
    2c88:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2c8c:	91248002 	add	x2, x0, #0x920
    2c90:	2a1303e1 	mov	w1, w19
    2c94:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2c98:	91252000 	add	x0, x0, #0x948
    2c9c:	94000cb7 	bl	5f78 <spt_ComplainError>
    2ca0:	97fff9c8 	bl	13c0 <__errno_location@plt>
    2ca4:	b9400000 	ldr	w0, [x0]
    2ca8:	11404000 	add	w0, w0, #0x10, lsl #12
    2cac:	1400013f 	b	31a8 <p_tt_read_file+0x654>
	memset(tsr->sortorder, 0, tsr->nmodes * sizeof tsr->sortorder[0]);
    2cb0:	f9401fe0 	ldr	x0, [sp, #56]
    2cb4:	f9400403 	ldr	x3, [x0, #8]
    2cb8:	f9401fe0 	ldr	x0, [sp, #56]
    2cbc:	b9400000 	ldr	w0, [x0]
    2cc0:	2a0003e0 	mov	w0, w0
    2cc4:	d37ef400 	lsl	x0, x0, #2
    2cc8:	aa0003e2 	mov	x2, x0
    2ccc:	52800001 	mov	w1, #0x0                   	// #0
    2cd0:	aa0303e0 	mov	x0, x3
    2cd4:	97fff967 	bl	1270 <memset@plt>
	tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    2cd8:	f9401fe0 	ldr	x0, [sp, #56]
    2cdc:	b9400000 	ldr	w0, [x0]
    2ce0:	2a0003e0 	mov	w0, w0
    2ce4:	d37ef400 	lsl	x0, x0, #2
    2ce8:	97fff94e 	bl	1220 <malloc@plt>
    2cec:	aa0003e1 	mov	x1, x0
    2cf0:	f9401fe0 	ldr	x0, [sp, #56]
    2cf4:	f9000801 	str	x1, [x0, #16]
	spt_CheckOSError(!tsr->ndims, "SpTns Load");
    2cf8:	f9401fe0 	ldr	x0, [sp, #56]
    2cfc:	f9400800 	ldr	x0, [x0, #16]
    2d00:	f100001f 	cmp	x0, #0x0
    2d04:	1a9f17e0 	cset	w0, eq  // eq = none
    2d08:	12001c00 	and	w0, w0, #0xff
    2d0c:	92401c00 	and	x0, x0, #0xff
    2d10:	f100001f 	cmp	x0, #0x0
    2d14:	54000360 	b.eq	2d80 <p_tt_read_file+0x22c>  // b.none
    2d18:	97fff9aa 	bl	13c0 <__errno_location@plt>
    2d1c:	b9400000 	ldr	w0, [x0]
    2d20:	3140401f 	cmn	w0, #0x10, lsl #12
    2d24:	1a9f07e0 	cset	w0, ne  // ne = any
    2d28:	12001c00 	and	w0, w0, #0xff
    2d2c:	92401c00 	and	x0, x0, #0xff
    2d30:	f100001f 	cmp	x0, #0x0
    2d34:	54000260 	b.eq	2d80 <p_tt_read_file+0x22c>  // b.none
    2d38:	97fff9a2 	bl	13c0 <__errno_location@plt>
    2d3c:	b9400000 	ldr	w0, [x0]
    2d40:	11404013 	add	w19, w0, #0x10, lsl #12
    2d44:	97fff99f 	bl	13c0 <__errno_location@plt>
    2d48:	b9400000 	ldr	w0, [x0]
    2d4c:	97fff95d 	bl	12c0 <strerror@plt>
    2d50:	aa0003e4 	mov	x4, x0
    2d54:	52800a03 	mov	w3, #0x50                  	// #80
    2d58:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2d5c:	91248002 	add	x2, x0, #0x920
    2d60:	2a1303e1 	mov	w1, w19
    2d64:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2d68:	91252000 	add	x0, x0, #0x948
    2d6c:	94000c83 	bl	5f78 <spt_ComplainError>
    2d70:	97fff994 	bl	13c0 <__errno_location@plt>
    2d74:	b9400000 	ldr	w0, [x0]
    2d78:	11404000 	add	w0, w0, #0x10, lsl #12
    2d7c:	1400010b 	b	31a8 <p_tt_read_file+0x654>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2d80:	b9005bff 	str	wzr, [sp, #88]
    2d84:	14000030 	b	2e44 <p_tt_read_file+0x2f0>
		iores = fscanf(fp, "%u", &tsr->ndims[mode]);
    2d88:	f9401fe0 	ldr	x0, [sp, #56]
    2d8c:	f9400801 	ldr	x1, [x0, #16]
    2d90:	b9405be0 	ldr	w0, [sp, #88]
    2d94:	d37ef400 	lsl	x0, x0, #2
    2d98:	8b000020 	add	x0, x1, x0
    2d9c:	aa0003e2 	mov	x2, x0
    2da0:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2da4:	91246001 	add	x1, x0, #0x918
    2da8:	f94017e0 	ldr	x0, [sp, #40]
    2dac:	97fff921 	bl	1230 <__isoc99_fscanf@plt>
    2db0:	b90057e0 	str	w0, [sp, #84]
		spt_CheckOSError(iores != 1, "SpTns Load");
    2db4:	b94057e0 	ldr	w0, [sp, #84]
    2db8:	7100041f 	cmp	w0, #0x1
    2dbc:	1a9f07e0 	cset	w0, ne  // ne = any
    2dc0:	12001c00 	and	w0, w0, #0xff
    2dc4:	92401c00 	and	x0, x0, #0xff
    2dc8:	f100001f 	cmp	x0, #0x0
    2dcc:	54000360 	b.eq	2e38 <p_tt_read_file+0x2e4>  // b.none
    2dd0:	97fff97c 	bl	13c0 <__errno_location@plt>
    2dd4:	b9400000 	ldr	w0, [x0]
    2dd8:	3140401f 	cmn	w0, #0x10, lsl #12
    2ddc:	1a9f07e0 	cset	w0, ne  // ne = any
    2de0:	12001c00 	and	w0, w0, #0xff
    2de4:	92401c00 	and	x0, x0, #0xff
    2de8:	f100001f 	cmp	x0, #0x0
    2dec:	54000260 	b.eq	2e38 <p_tt_read_file+0x2e4>  // b.none
    2df0:	97fff974 	bl	13c0 <__errno_location@plt>
    2df4:	b9400000 	ldr	w0, [x0]
    2df8:	11404013 	add	w19, w0, #0x10, lsl #12
    2dfc:	97fff971 	bl	13c0 <__errno_location@plt>
    2e00:	b9400000 	ldr	w0, [x0]
    2e04:	97fff92f 	bl	12c0 <strerror@plt>
    2e08:	aa0003e4 	mov	x4, x0
    2e0c:	52800a63 	mov	w3, #0x53                  	// #83
    2e10:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2e14:	91248002 	add	x2, x0, #0x920
    2e18:	2a1303e1 	mov	w1, w19
    2e1c:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2e20:	91252000 	add	x0, x0, #0x948
    2e24:	94000c55 	bl	5f78 <spt_ComplainError>
    2e28:	97fff966 	bl	13c0 <__errno_location@plt>
    2e2c:	b9400000 	ldr	w0, [x0]
    2e30:	11404000 	add	w0, w0, #0x10, lsl #12
    2e34:	140000dd 	b	31a8 <p_tt_read_file+0x654>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2e38:	b9405be0 	ldr	w0, [sp, #88]
    2e3c:	11000400 	add	w0, w0, #0x1
    2e40:	b9005be0 	str	w0, [sp, #88]
    2e44:	f9401fe0 	ldr	x0, [sp, #56]
    2e48:	b9400000 	ldr	w0, [x0]
    2e4c:	b9405be1 	ldr	w1, [sp, #88]
    2e50:	6b00003f 	cmp	w1, w0
    2e54:	54fff9a3 	b.cc	2d88 <p_tt_read_file+0x234>  // b.lo, b.ul, b.last
	}
	tsr->nnz = 0;
    2e58:	f9401fe0 	ldr	x0, [sp, #56]
    2e5c:	f9000c1f 	str	xzr, [x0, #24]
	tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    2e60:	f9401fe0 	ldr	x0, [sp, #56]
    2e64:	b9400000 	ldr	w0, [x0]
    2e68:	2a0003e1 	mov	w1, w0
    2e6c:	aa0103e0 	mov	x0, x1
    2e70:	d37ff800 	lsl	x0, x0, #1
    2e74:	8b010000 	add	x0, x0, x1
    2e78:	d37df000 	lsl	x0, x0, #3
    2e7c:	97fff8e9 	bl	1220 <malloc@plt>
    2e80:	aa0003e1 	mov	x1, x0
    2e84:	f9401fe0 	ldr	x0, [sp, #56]
    2e88:	f9001001 	str	x1, [x0, #32]
	spt_CheckOSError(!tsr->inds, "SpTns Load");
    2e8c:	f9401fe0 	ldr	x0, [sp, #56]
    2e90:	f9401000 	ldr	x0, [x0, #32]
    2e94:	f100001f 	cmp	x0, #0x0
    2e98:	1a9f17e0 	cset	w0, eq  // eq = none
    2e9c:	12001c00 	and	w0, w0, #0xff
    2ea0:	92401c00 	and	x0, x0, #0xff
    2ea4:	f100001f 	cmp	x0, #0x0
    2ea8:	54000360 	b.eq	2f14 <p_tt_read_file+0x3c0>  // b.none
    2eac:	97fff945 	bl	13c0 <__errno_location@plt>
    2eb0:	b9400000 	ldr	w0, [x0]
    2eb4:	3140401f 	cmn	w0, #0x10, lsl #12
    2eb8:	1a9f07e0 	cset	w0, ne  // ne = any
    2ebc:	12001c00 	and	w0, w0, #0xff
    2ec0:	92401c00 	and	x0, x0, #0xff
    2ec4:	f100001f 	cmp	x0, #0x0
    2ec8:	54000260 	b.eq	2f14 <p_tt_read_file+0x3c0>  // b.none
    2ecc:	97fff93d 	bl	13c0 <__errno_location@plt>
    2ed0:	b9400000 	ldr	w0, [x0]
    2ed4:	11404013 	add	w19, w0, #0x10, lsl #12
    2ed8:	97fff93a 	bl	13c0 <__errno_location@plt>
    2edc:	b9400000 	ldr	w0, [x0]
    2ee0:	97fff8f8 	bl	12c0 <strerror@plt>
    2ee4:	aa0003e4 	mov	x4, x0
    2ee8:	52800ae3 	mov	w3, #0x57                  	// #87
    2eec:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2ef0:	91248002 	add	x2, x0, #0x920
    2ef4:	2a1303e1 	mov	w1, w19
    2ef8:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2efc:	91252000 	add	x0, x0, #0x948
    2f00:	94000c1e 	bl	5f78 <spt_ComplainError>
    2f04:	97fff92f 	bl	13c0 <__errno_location@plt>
    2f08:	b9400000 	ldr	w0, [x0]
    2f0c:	11404000 	add	w0, w0, #0x10, lsl #12
    2f10:	140000a6 	b	31a8 <p_tt_read_file+0x654>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2f14:	b9005bff 	str	wzr, [sp, #88]
    2f18:	14000021 	b	2f9c <p_tt_read_file+0x448>
		retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    2f1c:	f9401fe0 	ldr	x0, [sp, #56]
    2f20:	f9401002 	ldr	x2, [x0, #32]
    2f24:	b9405be1 	ldr	w1, [sp, #88]
    2f28:	aa0103e0 	mov	x0, x1
    2f2c:	d37ff800 	lsl	x0, x0, #1
    2f30:	8b010000 	add	x0, x0, x1
    2f34:	d37df000 	lsl	x0, x0, #3
    2f38:	8b000040 	add	x0, x2, x0
    2f3c:	d2800002 	mov	x2, #0x0                   	// #0
    2f40:	d2800001 	mov	x1, #0x0                   	// #0
    2f44:	97fffddb 	bl	26b0 <sptNewIndexVector>
    2f48:	b9005fe0 	str	w0, [sp, #92]
		spt_CheckError(retval, "SpTns Load", NULL);
    2f4c:	b9405fe0 	ldr	w0, [sp, #92]
    2f50:	7100001f 	cmp	w0, #0x0
    2f54:	1a9f07e0 	cset	w0, ne  // ne = any
    2f58:	12001c00 	and	w0, w0, #0xff
    2f5c:	92401c00 	and	x0, x0, #0xff
    2f60:	f100001f 	cmp	x0, #0x0
    2f64:	54000160 	b.eq	2f90 <p_tt_read_file+0x43c>  // b.none
    2f68:	d2800004 	mov	x4, #0x0                   	// #0
    2f6c:	52800b43 	mov	w3, #0x5a                  	// #90
    2f70:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2f74:	91248002 	add	x2, x0, #0x920
    2f78:	b9405fe1 	ldr	w1, [sp, #92]
    2f7c:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2f80:	91252000 	add	x0, x0, #0x948
    2f84:	94000bfd 	bl	5f78 <spt_ComplainError>
    2f88:	b9405fe0 	ldr	w0, [sp, #92]
    2f8c:	14000087 	b	31a8 <p_tt_read_file+0x654>
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    2f90:	b9405be0 	ldr	w0, [sp, #88]
    2f94:	11000400 	add	w0, w0, #0x1
    2f98:	b9005be0 	str	w0, [sp, #88]
    2f9c:	f9401fe0 	ldr	x0, [sp, #56]
    2fa0:	b9400000 	ldr	w0, [x0]
    2fa4:	b9405be1 	ldr	w1, [sp, #88]
    2fa8:	6b00003f 	cmp	w1, w0
    2fac:	54fffb83 	b.cc	2f1c <p_tt_read_file+0x3c8>  // b.lo, b.ul, b.last
	}
	retval = sptNewValueVector(&tsr->values, 0, 0);
    2fb0:	f9401fe0 	ldr	x0, [sp, #56]
    2fb4:	9100a000 	add	x0, x0, #0x28
    2fb8:	d2800002 	mov	x2, #0x0                   	// #0
    2fbc:	d2800001 	mov	x1, #0x0                   	// #0
    2fc0:	97fffcb1 	bl	2284 <sptNewValueVector>
    2fc4:	b9005fe0 	str	w0, [sp, #92]
	spt_CheckError(retval, "SpTns Load", NULL);
    2fc8:	b9405fe0 	ldr	w0, [sp, #92]
    2fcc:	7100001f 	cmp	w0, #0x0
    2fd0:	1a9f07e0 	cset	w0, ne  // ne = any
    2fd4:	12001c00 	and	w0, w0, #0xff
    2fd8:	92401c00 	and	x0, x0, #0xff
    2fdc:	f100001f 	cmp	x0, #0x0
    2fe0:	54000b20 	b.eq	3144 <p_tt_read_file+0x5f0>  // b.none
    2fe4:	d2800004 	mov	x4, #0x0                   	// #0
    2fe8:	52800ba3 	mov	w3, #0x5d                  	// #93
    2fec:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2ff0:	91248002 	add	x2, x0, #0x920
    2ff4:	b9405fe1 	ldr	w1, [sp, #92]
    2ff8:	90000020 	adrp	x0, 6000 <spt_ComplainError+0x88>
    2ffc:	91252000 	add	x0, x0, #0x948
    3000:	94000bde 	bl	5f78 <spt_ComplainError>
    3004:	b9405fe0 	ldr	w0, [sp, #92]
    3008:	14000068 	b	31a8 <p_tt_read_file+0x654>
	while(retval == 0) {
		double value;
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    300c:	b9005bff 	str	wzr, [sp, #88]
    3010:	1400002e 	b	30c8 <p_tt_read_file+0x574>
			sptIndex index;
			iores = fscanf(fp, "%u", &index);
    3014:	910113e0 	add	x0, sp, #0x44
    3018:	aa0003e2 	mov	x2, x0
    301c:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3020:	91246001 	add	x1, x0, #0x918
    3024:	f94017e0 	ldr	x0, [sp, #40]
    3028:	97fff882 	bl	1230 <__isoc99_fscanf@plt>
    302c:	b90057e0 	str	w0, [sp, #84]
			if(iores != 1) {
    3030:	b94057e0 	ldr	w0, [sp, #84]
    3034:	7100041f 	cmp	w0, #0x1
    3038:	54000080 	b.eq	3048 <p_tt_read_file+0x4f4>  // b.none
				retval = -1;
    303c:	12800000 	mov	w0, #0xffffffff            	// #-1
    3040:	b9005fe0 	str	w0, [sp, #92]
    3044:	14000026 	b	30dc <p_tt_read_file+0x588>
				break;
			}
			if(index < start_index) {
    3048:	b94047e0 	ldr	w0, [sp, #68]
    304c:	b94037e1 	ldr	w1, [sp, #52]
    3050:	6b00003f 	cmp	w1, w0
    3054:	54000189 	b.ls	3084 <p_tt_read_file+0x530>  // b.plast
				spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
    3058:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    305c:	91256004 	add	x4, x0, #0x958
    3060:	52800d03 	mov	w3, #0x68                  	// #104
    3064:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3068:	91248002 	add	x2, x0, #0x920
    306c:	52800061 	mov	w1, #0x3                   	// #3
    3070:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3074:	91252000 	add	x0, x0, #0x948
    3078:	94000bc0 	bl	5f78 <spt_ComplainError>
    307c:	52800060 	mov	w0, #0x3                   	// #3
    3080:	1400004a 	b	31a8 <p_tt_read_file+0x654>
			}
			sptAppendIndexVector(&tsr->inds[mode], index-start_index);
    3084:	f9401fe0 	ldr	x0, [sp, #56]
    3088:	f9401002 	ldr	x2, [x0, #32]
    308c:	b9405be1 	ldr	w1, [sp, #88]
    3090:	aa0103e0 	mov	x0, x1
    3094:	d37ff800 	lsl	x0, x0, #1
    3098:	8b010000 	add	x0, x0, x1
    309c:	d37df000 	lsl	x0, x0, #3
    30a0:	8b000042 	add	x2, x2, x0
    30a4:	b94047e1 	ldr	w1, [sp, #68]
    30a8:	b94037e0 	ldr	w0, [sp, #52]
    30ac:	4b000020 	sub	w0, w1, w0
    30b0:	2a0003e1 	mov	w1, w0
    30b4:	aa0203e0 	mov	x0, x2
    30b8:	97fffdc9 	bl	27dc <sptAppendIndexVector>
		for(mode = 0; mode < tsr->nmodes; ++mode) {
    30bc:	b9405be0 	ldr	w0, [sp, #88]
    30c0:	11000400 	add	w0, w0, #0x1
    30c4:	b9005be0 	str	w0, [sp, #88]
    30c8:	f9401fe0 	ldr	x0, [sp, #56]
    30cc:	b9400000 	ldr	w0, [x0]
    30d0:	b9405be1 	ldr	w1, [sp, #88]
    30d4:	6b00003f 	cmp	w1, w0
    30d8:	54fff9e3 	b.cc	3014 <p_tt_read_file+0x4c0>  // b.lo, b.ul, b.last
		}
		if(retval == 0) {
    30dc:	b9405fe0 	ldr	w0, [sp, #92]
    30e0:	7100001f 	cmp	w0, #0x0
    30e4:	54000301 	b.ne	3144 <p_tt_read_file+0x5f0>  // b.any
			iores = fscanf(fp, "%lf", &value);
    30e8:	910123e0 	add	x0, sp, #0x48
    30ec:	aa0003e2 	mov	x2, x0
    30f0:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    30f4:	9125c001 	add	x1, x0, #0x970
    30f8:	f94017e0 	ldr	x0, [sp, #40]
    30fc:	97fff84d 	bl	1230 <__isoc99_fscanf@plt>
    3100:	b90057e0 	str	w0, [sp, #84]
			if(iores != 1) {
    3104:	b94057e0 	ldr	w0, [sp, #84]
    3108:	7100041f 	cmp	w0, #0x1
    310c:	54000080 	b.eq	311c <p_tt_read_file+0x5c8>  // b.none
				retval = -1;
    3110:	12800000 	mov	w0, #0xffffffff            	// #-1
    3114:	b9005fe0 	str	w0, [sp, #92]
    3118:	1400000e 	b	3150 <p_tt_read_file+0x5fc>
				break;
			}
			sptAppendValueVector(&tsr->values, value);
    311c:	f9401fe0 	ldr	x0, [sp, #56]
    3120:	9100a000 	add	x0, x0, #0x28
    3124:	fd4027e0 	ldr	d0, [sp, #72]
    3128:	1e624000 	fcvt	s0, d0
    312c:	97fffcb8 	bl	240c <sptAppendValueVector>
			++tsr->nnz;
    3130:	f9401fe0 	ldr	x0, [sp, #56]
    3134:	f9400c00 	ldr	x0, [x0, #24]
    3138:	91000401 	add	x1, x0, #0x1
    313c:	f9401fe0 	ldr	x0, [sp, #56]
    3140:	f9000c01 	str	x1, [x0, #24]
	while(retval == 0) {
    3144:	b9405fe0 	ldr	w0, [sp, #92]
    3148:	7100001f 	cmp	w0, #0x0
    314c:	54fff600 	b.eq	300c <p_tt_read_file+0x4b8>  // b.none
		}
	}
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    3150:	b9005bff 	str	wzr, [sp, #88]
    3154:	1400000f 	b	3190 <p_tt_read_file+0x63c>
		tsr->inds[mode].len = tsr->nnz;
    3158:	f9401fe0 	ldr	x0, [sp, #56]
    315c:	f9401002 	ldr	x2, [x0, #32]
    3160:	b9405be1 	ldr	w1, [sp, #88]
    3164:	aa0103e0 	mov	x0, x1
    3168:	d37ff800 	lsl	x0, x0, #1
    316c:	8b010000 	add	x0, x0, x1
    3170:	d37df000 	lsl	x0, x0, #3
    3174:	8b000040 	add	x0, x2, x0
    3178:	f9401fe1 	ldr	x1, [sp, #56]
    317c:	f9400c21 	ldr	x1, [x1, #24]
    3180:	f9000001 	str	x1, [x0]
	for(mode = 0; mode < tsr->nmodes; ++mode) {
    3184:	b9405be0 	ldr	w0, [sp, #88]
    3188:	11000400 	add	w0, w0, #0x1
    318c:	b9005be0 	str	w0, [sp, #88]
    3190:	f9401fe0 	ldr	x0, [sp, #56]
    3194:	b9400000 	ldr	w0, [x0]
    3198:	b9405be1 	ldr	w1, [sp, #88]
    319c:	6b00003f 	cmp	w1, w0
    31a0:	54fffdc3 	b.cc	3158 <p_tt_read_file+0x604>  // b.lo, b.ul, b.last
	}
	// sptSparseTensorCollectZeros(tsr);

	return 0;
    31a4:	52800000 	mov	w0, #0x0                   	// #0
}
    31a8:	f9400bf3 	ldr	x19, [sp, #16]
    31ac:	a8c67bfd 	ldp	x29, x30, [sp], #96
    31b0:	d65f03c0 	ret

00000000000031b4 <read_binary_header>:


static void read_binary_header(
		FILE * fin,
		bin_header * header)
{
    31b4:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    31b8:	910003fd 	mov	x29, sp
    31bc:	f9000fe0 	str	x0, [sp, #24]
    31c0:	f9000be1 	str	x1, [sp, #16]
	fread(&(header->magic), sizeof(header->magic), 1, fin);
    31c4:	f9400be0 	ldr	x0, [sp, #16]
    31c8:	f9400fe3 	ldr	x3, [sp, #24]
    31cc:	d2800022 	mov	x2, #0x1                   	// #1
    31d0:	d2800081 	mov	x1, #0x4                   	// #4
    31d4:	97fff85b 	bl	1340 <fread@plt>
	fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
    31d8:	f9400be0 	ldr	x0, [sp, #16]
    31dc:	91002000 	add	x0, x0, #0x8
    31e0:	f9400fe3 	ldr	x3, [sp, #24]
    31e4:	d2800022 	mov	x2, #0x1                   	// #1
    31e8:	d2800101 	mov	x1, #0x8                   	// #8
    31ec:	97fff855 	bl	1340 <fread@plt>
	fread(&(header->val_width), sizeof(header->val_width), 1, fin);
    31f0:	f9400be0 	ldr	x0, [sp, #16]
    31f4:	91004000 	add	x0, x0, #0x10
    31f8:	f9400fe3 	ldr	x3, [sp, #24]
    31fc:	d2800022 	mov	x2, #0x1                   	// #1
    3200:	d2800101 	mov	x1, #0x8                   	// #8
    3204:	97fff84f 	bl	1340 <fread@plt>

	if(header->idx_width > PASTA_INDEX_TYPEWIDTH / 8) {
    3208:	f9400be0 	ldr	x0, [sp, #16]
    320c:	f9400400 	ldr	x0, [x0, #8]
    3210:	f100101f 	cmp	x0, #0x4
    3214:	54000249 	b.ls	325c <read_binary_header+0xa8>  // b.plast
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    3218:	900000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    321c:	f947d800 	ldr	x0, [x0, #4016]
    3220:	f9400004 	ldr	x4, [x0]
										"Build with PASTA_INDEX_TYPEWIDTH %lu\n",
						header->idx_width * 8, header->idx_width * 8);
    3224:	f9400be0 	ldr	x0, [sp, #16]
    3228:	f9400400 	ldr	x0, [x0, #8]
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    322c:	d37df001 	lsl	x1, x0, #3
						header->idx_width * 8, header->idx_width * 8);
    3230:	f9400be0 	ldr	x0, [sp, #16]
    3234:	f9400400 	ldr	x0, [x0, #8]
		fprintf(stderr, "SPLATT: ERROR input has %lu-bit integers. "
    3238:	d37df000 	lsl	x0, x0, #3
    323c:	aa0003e3 	mov	x3, x0
    3240:	aa0103e2 	mov	x2, x1
    3244:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3248:	9125e001 	add	x1, x0, #0x978
    324c:	aa0403e0 	mov	x0, x4
    3250:	97fff868 	bl	13f0 <fprintf@plt>
		exit(-1);
    3254:	12800000 	mov	w0, #0xffffffff            	// #-1
    3258:	97fff7ca 	bl	1180 <exit@plt>
	}

	if(header->val_width > PASTA_VALUE_TYPEWIDTH / 8) {
    325c:	f9400be0 	ldr	x0, [sp, #16]
    3260:	f9400800 	ldr	x0, [x0, #16]
    3264:	f100101f 	cmp	x0, #0x4
    3268:	54000209 	b.ls	32a8 <read_binary_header+0xf4>  // b.plast
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    326c:	900000a0 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    3270:	f947d800 	ldr	x0, [x0, #4016]
    3274:	f9400004 	ldr	x4, [x0]
										"Build with PASTA_VALUE_TYPEWIDTH %lu for full precision\n",
						header->val_width * 8, header->val_width * 8);
    3278:	f9400be0 	ldr	x0, [sp, #16]
    327c:	f9400800 	ldr	x0, [x0, #16]
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    3280:	d37df001 	lsl	x1, x0, #3
						header->val_width * 8, header->val_width * 8);
    3284:	f9400be0 	ldr	x0, [sp, #16]
    3288:	f9400800 	ldr	x0, [x0, #16]
		fprintf(stderr, "SPLATT: WARNING input has %lu-bit floating-point values. "
    328c:	d37df000 	lsl	x0, x0, #3
    3290:	aa0003e3 	mov	x3, x0
    3294:	aa0103e2 	mov	x2, x1
    3298:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    329c:	91272001 	add	x1, x0, #0x9c8
    32a0:	aa0403e0 	mov	x0, x4
    32a4:	97fff853 	bl	13f0 <fprintf@plt>
	}
}
    32a8:	d503201f 	nop
    32ac:	a8c27bfd 	ldp	x29, x30, [sp], #32
    32b0:	d65f03c0 	ret

00000000000032b4 <fill_binary_idx>:
static void fill_binary_idx(
		sptIndex * const buffer,
		sptIndex const count,
		bin_header const * const header,
		FILE * fin)
{
    32b4:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    32b8:	910003fd 	mov	x29, sp
    32bc:	f90017e0 	str	x0, [sp, #40]
    32c0:	b90027e1 	str	w1, [sp, #36]
    32c4:	f9000fe2 	str	x2, [sp, #24]
    32c8:	f9000be3 	str	x3, [sp, #16]
	if(header->idx_width == sizeof(sptIndex)) {
    32cc:	f9400fe0 	ldr	x0, [sp, #24]
    32d0:	f9400400 	ldr	x0, [x0, #8]
    32d4:	f100101f 	cmp	x0, #0x4
    32d8:	54000101 	b.ne	32f8 <fill_binary_idx+0x44>  // b.any
		fread(buffer, sizeof(sptIndex), count, fin);
    32dc:	b94027e0 	ldr	w0, [sp, #36]
    32e0:	f9400be3 	ldr	x3, [sp, #16]
    32e4:	aa0003e2 	mov	x2, x0
    32e8:	d2800081 	mov	x1, #0x4                   	// #4
    32ec:	f94017e0 	ldr	x0, [sp, #40]
    32f0:	97fff814 	bl	1340 <fread@plt>
				buffer[n + i] = ubuf[i];
			}
		}
		free(ubuf);
	}
}
    32f4:	14000030 	b	33b4 <fill_binary_idx+0x100>
		sptIndex const BUF_LEN = 1024*1024;
    32f8:	52a00200 	mov	w0, #0x100000              	// #1048576
    32fc:	b9005be0 	str	w0, [sp, #88]
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    3300:	b9405be0 	ldr	w0, [sp, #88]
    3304:	d37ef400 	lsl	x0, x0, #2
    3308:	97fff7c6 	bl	1220 <malloc@plt>
    330c:	f9002be0 	str	x0, [sp, #80]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    3310:	b9005fff 	str	wzr, [sp, #92]
    3314:	14000022 	b	339c <fill_binary_idx+0xe8>
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    3318:	b94027e1 	ldr	w1, [sp, #36]
    331c:	b9405fe0 	ldr	w0, [sp, #92]
    3320:	4b000020 	sub	w0, w1, w0
    3324:	b9405be2 	ldr	w2, [sp, #88]
    3328:	b9405be1 	ldr	w1, [sp, #88]
    332c:	6b00005f 	cmp	w2, w0
    3330:	1a809020 	csel	w0, w1, w0, ls  // ls = plast
    3334:	b9004fe0 	str	w0, [sp, #76]
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    3338:	b9404fe0 	ldr	w0, [sp, #76]
    333c:	f9400be3 	ldr	x3, [sp, #16]
    3340:	aa0003e2 	mov	x2, x0
    3344:	d2800081 	mov	x1, #0x4                   	// #4
    3348:	f9402be0 	ldr	x0, [sp, #80]
    334c:	97fff7fd 	bl	1340 <fread@plt>
#pragma omp parallel for schedule(static)
    3350:	b9404fe0 	ldr	w0, [sp, #76]
    3354:	b90047e0 	str	w0, [sp, #68]
    3358:	b9405fe0 	ldr	w0, [sp, #92]
    335c:	b90043e0 	str	w0, [sp, #64]
    3360:	f9402be0 	ldr	x0, [sp, #80]
    3364:	f9001fe0 	str	x0, [sp, #56]
    3368:	f94017e0 	ldr	x0, [sp, #40]
    336c:	f9001be0 	str	x0, [sp, #48]
    3370:	9100c3e0 	add	x0, sp, #0x30
    3374:	52800003 	mov	w3, #0x0                   	// #0
    3378:	52800002 	mov	w2, #0x0                   	// #0
    337c:	aa0003e1 	mov	x1, x0
    3380:	90000000 	adrp	x0, 3000 <p_tt_read_file+0x4ac>
    3384:	9126b000 	add	x0, x0, #0x9ac
    3388:	97fff812 	bl	13d0 <GOMP_parallel@plt>
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    338c:	b9405fe1 	ldr	w1, [sp, #92]
    3390:	b9405be0 	ldr	w0, [sp, #88]
    3394:	0b000020 	add	w0, w1, w0
    3398:	b9005fe0 	str	w0, [sp, #92]
    339c:	b9405fe1 	ldr	w1, [sp, #92]
    33a0:	b94027e0 	ldr	w0, [sp, #36]
    33a4:	6b00003f 	cmp	w1, w0
    33a8:	54fffb83 	b.cc	3318 <fill_binary_idx+0x64>  // b.lo, b.ul, b.last
		free(ubuf);
    33ac:	f9402be0 	ldr	x0, [sp, #80]
    33b0:	97fff7e8 	bl	1350 <free@plt>
}
    33b4:	d503201f 	nop
    33b8:	a8c67bfd 	ldp	x29, x30, [sp], #96
    33bc:	d65f03c0 	ret

00000000000033c0 <fill_binary_nnzidx>:
static void fill_binary_nnzidx(
		sptNnzIndex * const buffer,
		sptIndex const count,
		bin_header const * const header,
		FILE * fin)
{
    33c0:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    33c4:	910003fd 	mov	x29, sp
    33c8:	f90017e0 	str	x0, [sp, #40]
    33cc:	b90027e1 	str	w1, [sp, #36]
    33d0:	f9000fe2 	str	x2, [sp, #24]
    33d4:	f9000be3 	str	x3, [sp, #16]
	if(header->idx_width == sizeof(sptNnzIndex)) {
    33d8:	f9400fe0 	ldr	x0, [sp, #24]
    33dc:	f9400400 	ldr	x0, [x0, #8]
    33e0:	f100201f 	cmp	x0, #0x8
    33e4:	54000101 	b.ne	3404 <fill_binary_nnzidx+0x44>  // b.any
		fread(buffer, sizeof(sptNnzIndex), count, fin);
    33e8:	b94027e0 	ldr	w0, [sp, #36]
    33ec:	f9400be3 	ldr	x3, [sp, #16]
    33f0:	aa0003e2 	mov	x2, x0
    33f4:	d2800101 	mov	x1, #0x8                   	// #8
    33f8:	f94017e0 	ldr	x0, [sp, #40]
    33fc:	97fff7d1 	bl	1340 <fread@plt>
				buffer[n + i] = ubuf[i];
			}
		}
		free(ubuf);
	}
}
    3400:	14000030 	b	34c0 <fill_binary_nnzidx+0x100>
		sptIndex const BUF_LEN = 1024*1024;
    3404:	52a00200 	mov	w0, #0x100000              	// #1048576
    3408:	b9005be0 	str	w0, [sp, #88]
		uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    340c:	b9405be0 	ldr	w0, [sp, #88]
    3410:	d37ef400 	lsl	x0, x0, #2
    3414:	97fff783 	bl	1220 <malloc@plt>
    3418:	f9002be0 	str	x0, [sp, #80]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    341c:	b9005fff 	str	wzr, [sp, #92]
    3420:	14000022 	b	34a8 <fill_binary_nnzidx+0xe8>
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    3424:	b94027e1 	ldr	w1, [sp, #36]
    3428:	b9405fe0 	ldr	w0, [sp, #92]
    342c:	4b000020 	sub	w0, w1, w0
    3430:	b9405be2 	ldr	w2, [sp, #88]
    3434:	b9405be1 	ldr	w1, [sp, #88]
    3438:	6b00005f 	cmp	w2, w0
    343c:	1a809020 	csel	w0, w1, w0, ls  // ls = plast
    3440:	b9004fe0 	str	w0, [sp, #76]
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    3444:	b9404fe0 	ldr	w0, [sp, #76]
    3448:	f9400be3 	ldr	x3, [sp, #16]
    344c:	aa0003e2 	mov	x2, x0
    3450:	d2800081 	mov	x1, #0x4                   	// #4
    3454:	f9402be0 	ldr	x0, [sp, #80]
    3458:	97fff7ba 	bl	1340 <fread@plt>
#pragma omp parallel for schedule(static)
    345c:	b9404fe0 	ldr	w0, [sp, #76]
    3460:	b90047e0 	str	w0, [sp, #68]
    3464:	b9405fe0 	ldr	w0, [sp, #92]
    3468:	b90043e0 	str	w0, [sp, #64]
    346c:	f9402be0 	ldr	x0, [sp, #80]
    3470:	f9001fe0 	str	x0, [sp, #56]
    3474:	f94017e0 	ldr	x0, [sp, #40]
    3478:	f9001be0 	str	x0, [sp, #48]
    347c:	9100c3e0 	add	x0, sp, #0x30
    3480:	52800003 	mov	w3, #0x0                   	// #0
    3484:	52800002 	mov	w2, #0x0                   	// #0
    3488:	aa0003e1 	mov	x1, x0
    348c:	90000000 	adrp	x0, 3000 <p_tt_read_file+0x4ac>
    3490:	912a9000 	add	x0, x0, #0xaa4
    3494:	97fff7cf 	bl	13d0 <GOMP_parallel@plt>
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    3498:	b9405fe1 	ldr	w1, [sp, #92]
    349c:	b9405be0 	ldr	w0, [sp, #88]
    34a0:	0b000020 	add	w0, w1, w0
    34a4:	b9005fe0 	str	w0, [sp, #92]
    34a8:	b9405fe1 	ldr	w1, [sp, #92]
    34ac:	b94027e0 	ldr	w0, [sp, #36]
    34b0:	6b00003f 	cmp	w1, w0
    34b4:	54fffb83 	b.cc	3424 <fill_binary_nnzidx+0x64>  // b.lo, b.ul, b.last
		free(ubuf);
    34b8:	f9402be0 	ldr	x0, [sp, #80]
    34bc:	97fff7a5 	bl	1350 <free@plt>
}
    34c0:	d503201f 	nop
    34c4:	a8c67bfd 	ldp	x29, x30, [sp], #96
    34c8:	d65f03c0 	ret

00000000000034cc <fill_binary_val>:
static void fill_binary_val(
		sptValue * const buffer,
		sptIndex const count,
		bin_header const * const header,
		FILE * fin)
{
    34cc:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    34d0:	910003fd 	mov	x29, sp
    34d4:	f90017e0 	str	x0, [sp, #40]
    34d8:	b90027e1 	str	w1, [sp, #36]
    34dc:	f9000fe2 	str	x2, [sp, #24]
    34e0:	f9000be3 	str	x3, [sp, #16]
	if(header->val_width == sizeof(sptValue)) {
    34e4:	f9400fe0 	ldr	x0, [sp, #24]
    34e8:	f9400800 	ldr	x0, [x0, #16]
    34ec:	f100101f 	cmp	x0, #0x4
    34f0:	54000101 	b.ne	3510 <fill_binary_val+0x44>  // b.any
		fread(buffer, sizeof(sptValue), count, fin);
    34f4:	b94027e0 	ldr	w0, [sp, #36]
    34f8:	f9400be3 	ldr	x3, [sp, #16]
    34fc:	aa0003e2 	mov	x2, x0
    3500:	d2800081 	mov	x1, #0x4                   	// #4
    3504:	f94017e0 	ldr	x0, [sp, #40]
    3508:	97fff78e 	bl	1340 <fread@plt>
				buffer[n + i] = ubuf[i];
			}
		}
		free(ubuf);
	}
}
    350c:	14000030 	b	35cc <fill_binary_val+0x100>
		sptIndex const BUF_LEN = 1024*1024;
    3510:	52a00200 	mov	w0, #0x100000              	// #1048576
    3514:	b9005be0 	str	w0, [sp, #88]
		double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
    3518:	b9405be0 	ldr	w0, [sp, #88]
    351c:	d37df000 	lsl	x0, x0, #3
    3520:	97fff740 	bl	1220 <malloc@plt>
    3524:	f9002be0 	str	x0, [sp, #80]
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    3528:	b9005fff 	str	wzr, [sp, #92]
    352c:	14000022 	b	35b4 <fill_binary_val+0xe8>
			sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
    3530:	b94027e1 	ldr	w1, [sp, #36]
    3534:	b9405fe0 	ldr	w0, [sp, #92]
    3538:	4b000020 	sub	w0, w1, w0
    353c:	b9405be2 	ldr	w2, [sp, #88]
    3540:	b9405be1 	ldr	w1, [sp, #88]
    3544:	6b00005f 	cmp	w2, w0
    3548:	1a809020 	csel	w0, w1, w0, ls  // ls = plast
    354c:	b9004fe0 	str	w0, [sp, #76]
			fread(ubuf, sizeof(*ubuf), read_count, fin);
    3550:	b9404fe0 	ldr	w0, [sp, #76]
    3554:	f9400be3 	ldr	x3, [sp, #16]
    3558:	aa0003e2 	mov	x2, x0
    355c:	d2800101 	mov	x1, #0x8                   	// #8
    3560:	f9402be0 	ldr	x0, [sp, #80]
    3564:	97fff777 	bl	1340 <fread@plt>
#pragma omp parallel for schedule(static)
    3568:	b9404fe0 	ldr	w0, [sp, #76]
    356c:	b90047e0 	str	w0, [sp, #68]
    3570:	b9405fe0 	ldr	w0, [sp, #92]
    3574:	b90043e0 	str	w0, [sp, #64]
    3578:	f9402be0 	ldr	x0, [sp, #80]
    357c:	f9001fe0 	str	x0, [sp, #56]
    3580:	f94017e0 	ldr	x0, [sp, #40]
    3584:	f9001be0 	str	x0, [sp, #48]
    3588:	9100c3e0 	add	x0, sp, #0x30
    358c:	52800003 	mov	w3, #0x0                   	// #0
    3590:	52800002 	mov	w2, #0x0                   	// #0
    3594:	aa0003e1 	mov	x1, x0
    3598:	90000000 	adrp	x0, 3000 <p_tt_read_file+0x4ac>
    359c:	912e8000 	add	x0, x0, #0xba0
    35a0:	97fff78c 	bl	13d0 <GOMP_parallel@plt>
		for(sptIndex n=0; n < count; n += BUF_LEN) {
    35a4:	b9405fe1 	ldr	w1, [sp, #92]
    35a8:	b9405be0 	ldr	w0, [sp, #88]
    35ac:	0b000020 	add	w0, w1, w0
    35b0:	b9005fe0 	str	w0, [sp, #92]
    35b4:	b9405fe1 	ldr	w1, [sp, #92]
    35b8:	b94027e0 	ldr	w0, [sp, #36]
    35bc:	6b00003f 	cmp	w1, w0
    35c0:	54fffb83 	b.cc	3530 <fill_binary_val+0x64>  // b.lo, b.ul, b.last
		free(ubuf);
    35c4:	f9402be0 	ldr	x0, [sp, #80]
    35c8:	97fff762 	bl	1350 <free@plt>
}
    35cc:	d503201f 	nop
    35d0:	a8c67bfd 	ldp	x29, x30, [sp], #96
    35d4:	d65f03c0 	ret

00000000000035d8 <p_tt_read_binary_file>:
* @param fin The file to read from.
*
* @return The parsed tensor.
*/
static int p_tt_read_binary_file(sptSparseTensor *tsr, FILE * fin)
{
    35d8:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    35dc:	910003fd 	mov	x29, sp
    35e0:	f9000fe0 	str	x0, [sp, #24]
    35e4:	f9000be1 	str	x1, [sp, #16]
	int result;
	bin_header header;
	read_binary_header(fin, &header);
    35e8:	9100c3e0 	add	x0, sp, #0x30
    35ec:	aa0003e1 	mov	x1, x0
    35f0:	f9400be0 	ldr	x0, [sp, #16]
    35f4:	97fffef0 	bl	31b4 <read_binary_header>
	// printf("header.magic: %d\n", header.magic);
	// printf("header.idx_width: %lu\n", header.idx_width);
	// printf("header.val_width: %lu\n", header.val_width);

	sptNnzIndex nnz = 0;
    35f8:	f90017ff 	str	xzr, [sp, #40]
	sptIndex nmodes = 0;
    35fc:	b90027ff 	str	wzr, [sp, #36]

	fill_binary_idx(&nmodes, 1, &header, fin);
    3600:	9100c3e1 	add	x1, sp, #0x30
    3604:	910093e0 	add	x0, sp, #0x24
    3608:	f9400be3 	ldr	x3, [sp, #16]
    360c:	aa0103e2 	mov	x2, x1
    3610:	52800021 	mov	w1, #0x1                   	// #1
    3614:	97ffff28 	bl	32b4 <fill_binary_idx>

	sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
    3618:	b94027e0 	ldr	w0, [sp, #36]
    361c:	2a0003e0 	mov	w0, w0
    3620:	d37ef400 	lsl	x0, x0, #2
    3624:	97fff6ff 	bl	1220 <malloc@plt>
    3628:	f9002be0 	str	x0, [sp, #80]
	fill_binary_idx(dims, nmodes, &header, fin);
    362c:	b94027e0 	ldr	w0, [sp, #36]
    3630:	9100c3e1 	add	x1, sp, #0x30
    3634:	f9400be3 	ldr	x3, [sp, #16]
    3638:	aa0103e2 	mov	x2, x1
    363c:	2a0003e1 	mov	w1, w0
    3640:	f9402be0 	ldr	x0, [sp, #80]
    3644:	97ffff1c 	bl	32b4 <fill_binary_idx>
	fill_binary_nnzidx(&nnz, 1, &header, fin);
    3648:	9100c3e1 	add	x1, sp, #0x30
    364c:	9100a3e0 	add	x0, sp, #0x28
    3650:	f9400be3 	ldr	x3, [sp, #16]
    3654:	aa0103e2 	mov	x2, x1
    3658:	52800021 	mov	w1, #0x1                   	// #1
    365c:	97ffff59 	bl	33c0 <fill_binary_nnzidx>

	/* allocate structures */
	sptNewSparseTensor(tsr, nmodes, dims);
    3660:	b94027e0 	ldr	w0, [sp, #36]
    3664:	f9402be2 	ldr	x2, [sp, #80]
    3668:	2a0003e1 	mov	w1, w0
    366c:	f9400fe0 	ldr	x0, [sp, #24]
    3670:	97fffa67 	bl	200c <sptNewSparseTensor>
	tsr->nnz = nnz;
    3674:	f94017e1 	ldr	x1, [sp, #40]
    3678:	f9400fe0 	ldr	x0, [sp, #24]
    367c:	f9000c01 	str	x1, [x0, #24]
	for(sptIndex m=0; m < nmodes; ++m) {
    3680:	b9005fff 	str	wzr, [sp, #92]
    3684:	14000020 	b	3704 <p_tt_read_binary_file+0x12c>
		result = sptResizeIndexVector(&tsr->inds[m], nnz);
    3688:	f9400fe0 	ldr	x0, [sp, #24]
    368c:	f9401002 	ldr	x2, [x0, #32]
    3690:	b9405fe1 	ldr	w1, [sp, #92]
    3694:	aa0103e0 	mov	x0, x1
    3698:	d37ff800 	lsl	x0, x0, #1
    369c:	8b010000 	add	x0, x0, x1
    36a0:	d37df000 	lsl	x0, x0, #3
    36a4:	8b000040 	add	x0, x2, x0
    36a8:	f94017e1 	ldr	x1, [sp, #40]
    36ac:	97fffc9e 	bl	2924 <sptResizeIndexVector>
    36b0:	b9004fe0 	str	w0, [sp, #76]
		spt_CheckError(result, "SpTns Read", NULL);
    36b4:	b9404fe0 	ldr	w0, [sp, #76]
    36b8:	7100001f 	cmp	w0, #0x0
    36bc:	1a9f07e0 	cset	w0, ne  // ne = any
    36c0:	12001c00 	and	w0, w0, #0xff
    36c4:	92401c00 	and	x0, x0, #0xff
    36c8:	f100001f 	cmp	x0, #0x0
    36cc:	54000160 	b.eq	36f8 <p_tt_read_binary_file+0x120>  // b.none
    36d0:	d2800004 	mov	x4, #0x0                   	// #0
    36d4:	528020a3 	mov	w3, #0x105                 	// #261
    36d8:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    36dc:	91248002 	add	x2, x0, #0x920
    36e0:	b9404fe1 	ldr	w1, [sp, #76]
    36e4:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    36e8:	91290000 	add	x0, x0, #0xa40
    36ec:	94000a23 	bl	5f78 <spt_ComplainError>
    36f0:	b9404fe0 	ldr	w0, [sp, #76]
    36f4:	14000041 	b	37f8 <p_tt_read_binary_file+0x220>
	for(sptIndex m=0; m < nmodes; ++m) {
    36f8:	b9405fe0 	ldr	w0, [sp, #92]
    36fc:	11000400 	add	w0, w0, #0x1
    3700:	b9005fe0 	str	w0, [sp, #92]
    3704:	b94027e0 	ldr	w0, [sp, #36]
    3708:	b9405fe1 	ldr	w1, [sp, #92]
    370c:	6b00003f 	cmp	w1, w0
    3710:	54fffbc3 	b.cc	3688 <p_tt_read_binary_file+0xb0>  // b.lo, b.ul, b.last
	}
	result = sptResizeValueVector(&tsr->values, nnz);
    3714:	f9400fe0 	ldr	x0, [sp, #24]
    3718:	9100a000 	add	x0, x0, #0x28
    371c:	f94017e1 	ldr	x1, [sp, #40]
    3720:	97fffb8d 	bl	2554 <sptResizeValueVector>
    3724:	b9004fe0 	str	w0, [sp, #76]
	spt_CheckError(result, "SpTns Read", NULL);
    3728:	b9404fe0 	ldr	w0, [sp, #76]
    372c:	7100001f 	cmp	w0, #0x0
    3730:	1a9f07e0 	cset	w0, ne  // ne = any
    3734:	12001c00 	and	w0, w0, #0xff
    3738:	92401c00 	and	x0, x0, #0xff
    373c:	f100001f 	cmp	x0, #0x0
    3740:	54000160 	b.eq	376c <p_tt_read_binary_file+0x194>  // b.none
    3744:	d2800004 	mov	x4, #0x0                   	// #0
    3748:	52802103 	mov	w3, #0x108                 	// #264
    374c:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3750:	91248002 	add	x2, x0, #0x920
    3754:	b9404fe1 	ldr	w1, [sp, #76]
    3758:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    375c:	91290000 	add	x0, x0, #0xa40
    3760:	94000a06 	bl	5f78 <spt_ComplainError>
    3764:	b9404fe0 	ldr	w0, [sp, #76]
    3768:	14000024 	b	37f8 <p_tt_read_binary_file+0x220>

	/* fill in tensor data */
	for(sptIndex m=0; m < nmodes; ++m) {
    376c:	b9005bff 	str	wzr, [sp, #88]
    3770:	14000014 	b	37c0 <p_tt_read_binary_file+0x1e8>
		fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
    3774:	f9400fe0 	ldr	x0, [sp, #24]
    3778:	f9401002 	ldr	x2, [x0, #32]
    377c:	b9405be1 	ldr	w1, [sp, #88]
    3780:	aa0103e0 	mov	x0, x1
    3784:	d37ff800 	lsl	x0, x0, #1
    3788:	8b010000 	add	x0, x0, x1
    378c:	d37df000 	lsl	x0, x0, #3
    3790:	8b000040 	add	x0, x2, x0
    3794:	f9400800 	ldr	x0, [x0, #16]
    3798:	f94017e1 	ldr	x1, [sp, #40]
    379c:	2a0103e4 	mov	w4, w1
    37a0:	9100c3e1 	add	x1, sp, #0x30
    37a4:	f9400be3 	ldr	x3, [sp, #16]
    37a8:	aa0103e2 	mov	x2, x1
    37ac:	2a0403e1 	mov	w1, w4
    37b0:	97fffec1 	bl	32b4 <fill_binary_idx>
	for(sptIndex m=0; m < nmodes; ++m) {
    37b4:	b9405be0 	ldr	w0, [sp, #88]
    37b8:	11000400 	add	w0, w0, #0x1
    37bc:	b9005be0 	str	w0, [sp, #88]
    37c0:	b94027e0 	ldr	w0, [sp, #36]
    37c4:	b9405be1 	ldr	w1, [sp, #88]
    37c8:	6b00003f 	cmp	w1, w0
    37cc:	54fffd43 	b.cc	3774 <p_tt_read_binary_file+0x19c>  // b.lo, b.ul, b.last
	}
	fill_binary_val(tsr->values.data, nnz, &header, fin);
    37d0:	f9400fe0 	ldr	x0, [sp, #24]
    37d4:	f9401c00 	ldr	x0, [x0, #56]
    37d8:	f94017e1 	ldr	x1, [sp, #40]
    37dc:	2a0103e4 	mov	w4, w1
    37e0:	9100c3e1 	add	x1, sp, #0x30
    37e4:	f9400be3 	ldr	x3, [sp, #16]
    37e8:	aa0103e2 	mov	x2, x1
    37ec:	2a0403e1 	mov	w1, w4
    37f0:	97ffff37 	bl	34cc <fill_binary_val>

	return 0;
    37f4:	52800000 	mov	w0, #0x0                   	// #0
}
    37f8:	a8c67bfd 	ldp	x29, x30, [sp], #96
    37fc:	d65f03c0 	ret

0000000000003800 <sptLoadSparseTensor>:
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname)
{
    3800:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    3804:	910003fd 	mov	x29, sp
    3808:	f9000bf3 	str	x19, [sp, #16]
    380c:	f9001fe0 	str	x0, [sp, #56]
    3810:	b90037e1 	str	w1, [sp, #52]
    3814:	f90017e2 	str	x2, [sp, #40]
	FILE * fp = fopen(fname, "r");
    3818:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    381c:	91294001 	add	x1, x0, #0xa50
    3820:	f94017e0 	ldr	x0, [sp, #40]
    3824:	97fff677 	bl	1200 <fopen@plt>
    3828:	f90027e0 	str	x0, [sp, #72]
	sptAssert(fp != NULL);
    382c:	f94027e0 	ldr	x0, [sp, #72]
    3830:	f100001f 	cmp	x0, #0x0
    3834:	54000061 	b.ne	3840 <sptLoadSparseTensor+0x40>  // b.any
    3838:	12800000 	mov	w0, #0xffffffff            	// #-1
    383c:	97fff651 	bl	1180 <exit@plt>

	int iores;
	switch(get_file_type(fname)) {
    3840:	f94017e0 	ldr	x0, [sp, #40]
    3844:	97fffc8f 	bl	2a80 <get_file_type>
    3848:	7100001f 	cmp	w0, #0x0
    384c:	54000080 	b.eq	385c <sptLoadSparseTensor+0x5c>  // b.none
    3850:	7100041f 	cmp	w0, #0x1
    3854:	54000500 	b.eq	38f4 <sptLoadSparseTensor+0xf4>  // b.none
    3858:	1400004f 	b	3994 <sptLoadSparseTensor+0x194>
		case 0:
			iores = p_tt_read_file(tsr, start_index, fp);
    385c:	f94027e2 	ldr	x2, [sp, #72]
    3860:	b94037e1 	ldr	w1, [sp, #52]
    3864:	f9401fe0 	ldr	x0, [sp, #56]
    3868:	97fffcbb 	bl	2b54 <p_tt_read_file>
    386c:	b90047e0 	str	w0, [sp, #68]
			spt_CheckOSError(iores != 0, "SpTns Load");
    3870:	b94047e0 	ldr	w0, [sp, #68]
    3874:	7100001f 	cmp	w0, #0x0
    3878:	1a9f07e0 	cset	w0, ne  // ne = any
    387c:	12001c00 	and	w0, w0, #0xff
    3880:	92401c00 	and	x0, x0, #0xff
    3884:	f100001f 	cmp	x0, #0x0
    3888:	54000800 	b.eq	3988 <sptLoadSparseTensor+0x188>  // b.none
    388c:	97fff6cd 	bl	13c0 <__errno_location@plt>
    3890:	b9400000 	ldr	w0, [x0]
    3894:	3140401f 	cmn	w0, #0x10, lsl #12
    3898:	1a9f07e0 	cset	w0, ne  // ne = any
    389c:	12001c00 	and	w0, w0, #0xff
    38a0:	92401c00 	and	x0, x0, #0xff
    38a4:	f100001f 	cmp	x0, #0x0
    38a8:	54000700 	b.eq	3988 <sptLoadSparseTensor+0x188>  // b.none
    38ac:	97fff6c5 	bl	13c0 <__errno_location@plt>
    38b0:	b9400000 	ldr	w0, [x0]
    38b4:	11404013 	add	w19, w0, #0x10, lsl #12
    38b8:	97fff6c2 	bl	13c0 <__errno_location@plt>
    38bc:	b9400000 	ldr	w0, [x0]
    38c0:	97fff680 	bl	12c0 <strerror@plt>
    38c4:	aa0003e4 	mov	x4, x0
    38c8:	52802483 	mov	w3, #0x124                 	// #292
    38cc:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    38d0:	91248002 	add	x2, x0, #0x920
    38d4:	2a1303e1 	mov	w1, w19
    38d8:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    38dc:	91252000 	add	x0, x0, #0x948
    38e0:	940009a6 	bl	5f78 <spt_ComplainError>
    38e4:	97fff6b7 	bl	13c0 <__errno_location@plt>
    38e8:	b9400000 	ldr	w0, [x0]
    38ec:	11404000 	add	w0, w0, #0x10, lsl #12
    38f0:	1400002c 	b	39a0 <sptLoadSparseTensor+0x1a0>
			break;
		case 1:
			iores = p_tt_read_binary_file(tsr, fp);
    38f4:	f94027e1 	ldr	x1, [sp, #72]
    38f8:	f9401fe0 	ldr	x0, [sp, #56]
    38fc:	97ffff37 	bl	35d8 <p_tt_read_binary_file>
    3900:	b90047e0 	str	w0, [sp, #68]
			spt_CheckOSError(iores != 0, "SpTns Load");
    3904:	b94047e0 	ldr	w0, [sp, #68]
    3908:	7100001f 	cmp	w0, #0x0
    390c:	1a9f07e0 	cset	w0, ne  // ne = any
    3910:	12001c00 	and	w0, w0, #0xff
    3914:	92401c00 	and	x0, x0, #0xff
    3918:	f100001f 	cmp	x0, #0x0
    391c:	540003a0 	b.eq	3990 <sptLoadSparseTensor+0x190>  // b.none
    3920:	97fff6a8 	bl	13c0 <__errno_location@plt>
    3924:	b9400000 	ldr	w0, [x0]
    3928:	3140401f 	cmn	w0, #0x10, lsl #12
    392c:	1a9f07e0 	cset	w0, ne  // ne = any
    3930:	12001c00 	and	w0, w0, #0xff
    3934:	92401c00 	and	x0, x0, #0xff
    3938:	f100001f 	cmp	x0, #0x0
    393c:	540002a0 	b.eq	3990 <sptLoadSparseTensor+0x190>  // b.none
    3940:	97fff6a0 	bl	13c0 <__errno_location@plt>
    3944:	b9400000 	ldr	w0, [x0]
    3948:	11404013 	add	w19, w0, #0x10, lsl #12
    394c:	97fff69d 	bl	13c0 <__errno_location@plt>
    3950:	b9400000 	ldr	w0, [x0]
    3954:	97fff65b 	bl	12c0 <strerror@plt>
    3958:	aa0003e4 	mov	x4, x0
    395c:	52802503 	mov	w3, #0x128                 	// #296
    3960:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3964:	91248002 	add	x2, x0, #0x920
    3968:	2a1303e1 	mov	w1, w19
    396c:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3970:	91252000 	add	x0, x0, #0x948
    3974:	94000981 	bl	5f78 <spt_ComplainError>
    3978:	97fff692 	bl	13c0 <__errno_location@plt>
    397c:	b9400000 	ldr	w0, [x0]
    3980:	11404000 	add	w0, w0, #0x10, lsl #12
    3984:	14000007 	b	39a0 <sptLoadSparseTensor+0x1a0>
			break;
    3988:	d503201f 	nop
    398c:	14000002 	b	3994 <sptLoadSparseTensor+0x194>
			break;
    3990:	d503201f 	nop
	}

	fclose(fp);
    3994:	f94027e0 	ldr	x0, [sp, #72]
    3998:	97fff616 	bl	11f0 <fclose@plt>

	return 0;
    399c:	52800000 	mov	w0, #0x0                   	// #0
}
    39a0:	f9400bf3 	ldr	x19, [sp, #16]
    39a4:	a8c57bfd 	ldp	x29, x30, [sp], #80
    39a8:	d65f03c0 	ret

00000000000039ac <fill_binary_idx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    39ac:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    39b0:	910003fd 	mov	x29, sp
    39b4:	a90153f3 	stp	x19, x20, [sp, #16]
    39b8:	f90017e0 	str	x0, [sp, #40]
    39bc:	f94017e0 	ldr	x0, [sp, #40]
    39c0:	b9401400 	ldr	w0, [x0, #20]
    39c4:	b9004be0 	str	w0, [sp, #72]
    39c8:	f94017e0 	ldr	x0, [sp, #40]
    39cc:	b9401000 	ldr	w0, [x0, #16]
    39d0:	b90047e0 	str	w0, [sp, #68]
    39d4:	f94017e0 	ldr	x0, [sp, #40]
    39d8:	f9400400 	ldr	x0, [x0, #8]
    39dc:	f9001fe0 	str	x0, [sp, #56]
    39e0:	f94017e0 	ldr	x0, [sp, #40]
    39e4:	f9400000 	ldr	x0, [x0]
    39e8:	f9001be0 	str	x0, [sp, #48]
    39ec:	b9404bf3 	ldr	w19, [sp, #72]
    39f0:	7100027f 	cmp	w19, #0x0
    39f4:	54000500 	b.eq	3a94 <fill_binary_idx._omp_fn.0+0xe8>  // b.none
    39f8:	97fff62e 	bl	12b0 <omp_get_num_threads@plt>
    39fc:	2a0003f4 	mov	w20, w0
    3a00:	97fff5f8 	bl	11e0 <omp_get_thread_num@plt>
    3a04:	2a0003e2 	mov	w2, w0
    3a08:	1ad40a60 	udiv	w0, w19, w20
    3a0c:	1ad40a61 	udiv	w1, w19, w20
    3a10:	1b147c21 	mul	w1, w1, w20
    3a14:	4b010261 	sub	w1, w19, w1
    3a18:	6b01005f 	cmp	w2, w1
    3a1c:	54000363 	b.cc	3a88 <fill_binary_idx._omp_fn.0+0xdc>  // b.lo, b.ul, b.last
    3a20:	1b027c02 	mul	w2, w0, w2
    3a24:	0b010041 	add	w1, w2, w1
    3a28:	0b000020 	add	w0, w1, w0
    3a2c:	6b00003f 	cmp	w1, w0
    3a30:	54000322 	b.cs	3a94 <fill_binary_idx._omp_fn.0+0xe8>  // b.hs, b.nlast
    3a34:	b9004fe1 	str	w1, [sp, #76]
				buffer[n + i] = ubuf[i];
    3a38:	b9404fe1 	ldr	w1, [sp, #76]
    3a3c:	d37ef421 	lsl	x1, x1, #2
    3a40:	f9401fe2 	ldr	x2, [sp, #56]
    3a44:	8b010042 	add	x2, x2, x1
    3a48:	b94047e3 	ldr	w3, [sp, #68]
    3a4c:	b9404fe1 	ldr	w1, [sp, #76]
    3a50:	0b010061 	add	w1, w3, w1
    3a54:	2a0103e1 	mov	w1, w1
    3a58:	d37ef421 	lsl	x1, x1, #2
    3a5c:	f9401be3 	ldr	x3, [sp, #48]
    3a60:	8b010061 	add	x1, x3, x1
    3a64:	b9400042 	ldr	w2, [x2]
    3a68:	b9000022 	str	w2, [x1]
    3a6c:	b9404fe1 	ldr	w1, [sp, #76]
    3a70:	11000421 	add	w1, w1, #0x1
    3a74:	b9004fe1 	str	w1, [sp, #76]
    3a78:	b9404fe1 	ldr	w1, [sp, #76]
    3a7c:	6b00003f 	cmp	w1, w0
    3a80:	54fffdc3 	b.cc	3a38 <fill_binary_idx._omp_fn.0+0x8c>  // b.lo, b.ul, b.last
#pragma omp parallel for schedule(static)
    3a84:	14000004 	b	3a94 <fill_binary_idx._omp_fn.0+0xe8>
    3a88:	52800001 	mov	w1, #0x0                   	// #0
    3a8c:	11000400 	add	w0, w0, #0x1
    3a90:	17ffffe4 	b	3a20 <fill_binary_idx._omp_fn.0+0x74>
    3a94:	d503201f 	nop
    3a98:	a94153f3 	ldp	x19, x20, [sp, #16]
    3a9c:	a8c57bfd 	ldp	x29, x30, [sp], #80
    3aa0:	d65f03c0 	ret

0000000000003aa4 <fill_binary_nnzidx._omp_fn.0>:
#pragma omp parallel for schedule(static)
    3aa4:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    3aa8:	910003fd 	mov	x29, sp
    3aac:	a90153f3 	stp	x19, x20, [sp, #16]
    3ab0:	f90017e0 	str	x0, [sp, #40]
    3ab4:	f94017e0 	ldr	x0, [sp, #40]
    3ab8:	b9401400 	ldr	w0, [x0, #20]
    3abc:	b9004be0 	str	w0, [sp, #72]
    3ac0:	f94017e0 	ldr	x0, [sp, #40]
    3ac4:	b9401000 	ldr	w0, [x0, #16]
    3ac8:	b90047e0 	str	w0, [sp, #68]
    3acc:	f94017e0 	ldr	x0, [sp, #40]
    3ad0:	f9400400 	ldr	x0, [x0, #8]
    3ad4:	f9001fe0 	str	x0, [sp, #56]
    3ad8:	f94017e0 	ldr	x0, [sp, #40]
    3adc:	f9400000 	ldr	x0, [x0]
    3ae0:	f9001be0 	str	x0, [sp, #48]
    3ae4:	b9404bf3 	ldr	w19, [sp, #72]
    3ae8:	7100027f 	cmp	w19, #0x0
    3aec:	54000520 	b.eq	3b90 <fill_binary_nnzidx._omp_fn.0+0xec>  // b.none
    3af0:	97fff5f0 	bl	12b0 <omp_get_num_threads@plt>
    3af4:	2a0003f4 	mov	w20, w0
    3af8:	97fff5ba 	bl	11e0 <omp_get_thread_num@plt>
    3afc:	2a0003e2 	mov	w2, w0
    3b00:	1ad40a60 	udiv	w0, w19, w20
    3b04:	1ad40a61 	udiv	w1, w19, w20
    3b08:	1b147c21 	mul	w1, w1, w20
    3b0c:	4b010261 	sub	w1, w19, w1
    3b10:	6b01005f 	cmp	w2, w1
    3b14:	54000383 	b.cc	3b84 <fill_binary_nnzidx._omp_fn.0+0xe0>  // b.lo, b.ul, b.last
    3b18:	1b027c02 	mul	w2, w0, w2
    3b1c:	0b010041 	add	w1, w2, w1
    3b20:	0b000020 	add	w0, w1, w0
    3b24:	6b00003f 	cmp	w1, w0
    3b28:	54000342 	b.cs	3b90 <fill_binary_nnzidx._omp_fn.0+0xec>  // b.hs, b.nlast
    3b2c:	b9004fe1 	str	w1, [sp, #76]
				buffer[n + i] = ubuf[i];
    3b30:	b9404fe1 	ldr	w1, [sp, #76]
    3b34:	d37ef421 	lsl	x1, x1, #2
    3b38:	f9401fe2 	ldr	x2, [sp, #56]
    3b3c:	8b010041 	add	x1, x2, x1
    3b40:	b9400023 	ldr	w3, [x1]
    3b44:	b94047e2 	ldr	w2, [sp, #68]
    3b48:	b9404fe1 	ldr	w1, [sp, #76]
    3b4c:	0b010041 	add	w1, w2, w1
    3b50:	2a0103e1 	mov	w1, w1
    3b54:	d37df021 	lsl	x1, x1, #3
    3b58:	f9401be2 	ldr	x2, [sp, #48]
    3b5c:	8b010041 	add	x1, x2, x1
    3b60:	2a0303e2 	mov	w2, w3
    3b64:	f9000022 	str	x2, [x1]
    3b68:	b9404fe1 	ldr	w1, [sp, #76]
    3b6c:	11000421 	add	w1, w1, #0x1
    3b70:	b9004fe1 	str	w1, [sp, #76]
    3b74:	b9404fe1 	ldr	w1, [sp, #76]
    3b78:	6b00003f 	cmp	w1, w0
    3b7c:	54fffda3 	b.cc	3b30 <fill_binary_nnzidx._omp_fn.0+0x8c>  // b.lo, b.ul, b.last
#pragma omp parallel for schedule(static)
    3b80:	14000004 	b	3b90 <fill_binary_nnzidx._omp_fn.0+0xec>
    3b84:	52800001 	mov	w1, #0x0                   	// #0
    3b88:	11000400 	add	w0, w0, #0x1
    3b8c:	17ffffe3 	b	3b18 <fill_binary_nnzidx._omp_fn.0+0x74>
    3b90:	d503201f 	nop
    3b94:	a94153f3 	ldp	x19, x20, [sp, #16]
    3b98:	a8c57bfd 	ldp	x29, x30, [sp], #80
    3b9c:	d65f03c0 	ret

0000000000003ba0 <fill_binary_val._omp_fn.0>:
#pragma omp parallel for schedule(static)
    3ba0:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    3ba4:	910003fd 	mov	x29, sp
    3ba8:	a90153f3 	stp	x19, x20, [sp, #16]
    3bac:	f90017e0 	str	x0, [sp, #40]
    3bb0:	f94017e0 	ldr	x0, [sp, #40]
    3bb4:	b9401400 	ldr	w0, [x0, #20]
    3bb8:	b9004be0 	str	w0, [sp, #72]
    3bbc:	f94017e0 	ldr	x0, [sp, #40]
    3bc0:	b9401000 	ldr	w0, [x0, #16]
    3bc4:	b90047e0 	str	w0, [sp, #68]
    3bc8:	f94017e0 	ldr	x0, [sp, #40]
    3bcc:	f9400400 	ldr	x0, [x0, #8]
    3bd0:	f9001fe0 	str	x0, [sp, #56]
    3bd4:	f94017e0 	ldr	x0, [sp, #40]
    3bd8:	f9400000 	ldr	x0, [x0]
    3bdc:	f9001be0 	str	x0, [sp, #48]
    3be0:	b9404bf3 	ldr	w19, [sp, #72]
    3be4:	7100027f 	cmp	w19, #0x0
    3be8:	54000520 	b.eq	3c8c <fill_binary_val._omp_fn.0+0xec>  // b.none
    3bec:	97fff5b1 	bl	12b0 <omp_get_num_threads@plt>
    3bf0:	2a0003f4 	mov	w20, w0
    3bf4:	97fff57b 	bl	11e0 <omp_get_thread_num@plt>
    3bf8:	2a0003e2 	mov	w2, w0
    3bfc:	1ad40a60 	udiv	w0, w19, w20
    3c00:	1ad40a61 	udiv	w1, w19, w20
    3c04:	1b147c21 	mul	w1, w1, w20
    3c08:	4b010261 	sub	w1, w19, w1
    3c0c:	6b01005f 	cmp	w2, w1
    3c10:	54000383 	b.cc	3c80 <fill_binary_val._omp_fn.0+0xe0>  // b.lo, b.ul, b.last
    3c14:	1b027c02 	mul	w2, w0, w2
    3c18:	0b010041 	add	w1, w2, w1
    3c1c:	0b000020 	add	w0, w1, w0
    3c20:	6b00003f 	cmp	w1, w0
    3c24:	54000342 	b.cs	3c8c <fill_binary_val._omp_fn.0+0xec>  // b.hs, b.nlast
    3c28:	b9004fe1 	str	w1, [sp, #76]
				buffer[n + i] = ubuf[i];
    3c2c:	b9404fe1 	ldr	w1, [sp, #76]
    3c30:	d37df021 	lsl	x1, x1, #3
    3c34:	f9401fe2 	ldr	x2, [sp, #56]
    3c38:	8b010041 	add	x1, x2, x1
    3c3c:	fd400020 	ldr	d0, [x1]
    3c40:	b94047e2 	ldr	w2, [sp, #68]
    3c44:	b9404fe1 	ldr	w1, [sp, #76]
    3c48:	0b010041 	add	w1, w2, w1
    3c4c:	2a0103e1 	mov	w1, w1
    3c50:	d37ef421 	lsl	x1, x1, #2
    3c54:	f9401be2 	ldr	x2, [sp, #48]
    3c58:	8b010041 	add	x1, x2, x1
    3c5c:	1e624000 	fcvt	s0, d0
    3c60:	bd000020 	str	s0, [x1]
    3c64:	b9404fe1 	ldr	w1, [sp, #76]
    3c68:	11000421 	add	w1, w1, #0x1
    3c6c:	b9004fe1 	str	w1, [sp, #76]
    3c70:	b9404fe1 	ldr	w1, [sp, #76]
    3c74:	6b00003f 	cmp	w1, w0
    3c78:	54fffda3 	b.cc	3c2c <fill_binary_val._omp_fn.0+0x8c>  // b.lo, b.ul, b.last
#pragma omp parallel for schedule(static)
    3c7c:	14000004 	b	3c8c <fill_binary_val._omp_fn.0+0xec>
    3c80:	52800001 	mov	w1, #0x0                   	// #0
    3c84:	11000400 	add	w0, w0, #0x1
    3c88:	17ffffe3 	b	3c14 <fill_binary_val._omp_fn.0+0x74>
    3c8c:	d503201f 	nop
    3c90:	a94153f3 	ldp	x19, x20, [sp, #16]
    3c94:	a8c57bfd 	ldp	x29, x30, [sp], #80
    3c98:	d65f03c0 	ret

0000000000003c9c <sptNewMatrix>:
 * @param ncols the number of columns
 *
 * The memory layout of this dense matrix is a flat 2D array, with `ncols`
 * rounded up to multiples of 8
 */
int sptNewMatrix(sptMatrix *mtx, sptIndex const nrows, sptIndex const ncols) {
    3c9c:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    3ca0:	910003fd 	mov	x29, sp
    3ca4:	f9000bf3 	str	x19, [sp, #16]
    3ca8:	f90017e0 	str	x0, [sp, #40]
    3cac:	b90027e1 	str	w1, [sp, #36]
    3cb0:	b90023e2 	str	w2, [sp, #32]
	mtx->nrows = nrows;
    3cb4:	f94017e0 	ldr	x0, [sp, #40]
    3cb8:	b94027e1 	ldr	w1, [sp, #36]
    3cbc:	b9000001 	str	w1, [x0]
	mtx->ncols = ncols;
    3cc0:	f94017e0 	ldr	x0, [sp, #40]
    3cc4:	b94023e1 	ldr	w1, [sp, #32]
    3cc8:	b9000401 	str	w1, [x0, #4]
	mtx->cap = nrows != 0 ? nrows : 1;
    3ccc:	b94027e0 	ldr	w0, [sp, #36]
    3cd0:	7100001f 	cmp	w0, #0x0
    3cd4:	54000060 	b.eq	3ce0 <sptNewMatrix+0x44>  // b.none
    3cd8:	b94027e0 	ldr	w0, [sp, #36]
    3cdc:	14000002 	b	3ce4 <sptNewMatrix+0x48>
    3ce0:	52800020 	mov	w0, #0x1                   	// #1
    3ce4:	f94017e1 	ldr	x1, [sp, #40]
    3ce8:	b9000820 	str	w0, [x1, #8]
	mtx->stride = ((ncols-1)/8+1)*8;
    3cec:	b94023e0 	ldr	w0, [sp, #32]
    3cf0:	51000400 	sub	w0, w0, #0x1
    3cf4:	53037c00 	lsr	w0, w0, #3
    3cf8:	11000400 	add	w0, w0, #0x1
    3cfc:	531d7001 	lsl	w1, w0, #3
    3d00:	f94017e0 	ldr	x0, [sp, #40]
    3d04:	b9000c01 	str	w1, [x0, #12]
#ifdef _ISOC11_SOURCE
	mtx->values = aligned_alloc(8 * sizeof (sptValue), mtx->cap * mtx->stride * sizeof (sptValue));
    3d08:	f94017e0 	ldr	x0, [sp, #40]
    3d0c:	b9400801 	ldr	w1, [x0, #8]
    3d10:	f94017e0 	ldr	x0, [sp, #40]
    3d14:	b9400c00 	ldr	w0, [x0, #12]
    3d18:	1b007c20 	mul	w0, w1, w0
    3d1c:	2a0003e0 	mov	w0, w0
    3d20:	d37ef400 	lsl	x0, x0, #2
    3d24:	aa0003e1 	mov	x1, x0
    3d28:	d2800400 	mov	x0, #0x20                  	// #32
    3d2c:	97fff54d 	bl	1260 <aligned_alloc@plt>
    3d30:	aa0003e1 	mov	x1, x0
    3d34:	f94017e0 	ldr	x0, [sp, #40]
    3d38:	f9000801 	str	x1, [x0, #16]
		}
	}
#else
	mtx->values = malloc(mtx->cap * mtx->stride * sizeof (sptValue));
#endif
	spt_CheckOSError(!mtx->values, "Mtx New");
    3d3c:	f94017e0 	ldr	x0, [sp, #40]
    3d40:	f9400800 	ldr	x0, [x0, #16]
    3d44:	f100001f 	cmp	x0, #0x0
    3d48:	1a9f17e0 	cset	w0, eq  // eq = none
    3d4c:	12001c00 	and	w0, w0, #0xff
    3d50:	92401c00 	and	x0, x0, #0xff
    3d54:	f100001f 	cmp	x0, #0x0
    3d58:	54000360 	b.eq	3dc4 <sptNewMatrix+0x128>  // b.none
    3d5c:	97fff599 	bl	13c0 <__errno_location@plt>
    3d60:	b9400000 	ldr	w0, [x0]
    3d64:	3140401f 	cmn	w0, #0x10, lsl #12
    3d68:	1a9f07e0 	cset	w0, ne  // ne = any
    3d6c:	12001c00 	and	w0, w0, #0xff
    3d70:	92401c00 	and	x0, x0, #0xff
    3d74:	f100001f 	cmp	x0, #0x0
    3d78:	54000260 	b.eq	3dc4 <sptNewMatrix+0x128>  // b.none
    3d7c:	97fff591 	bl	13c0 <__errno_location@plt>
    3d80:	b9400000 	ldr	w0, [x0]
    3d84:	11404013 	add	w19, w0, #0x10, lsl #12
    3d88:	97fff58e 	bl	13c0 <__errno_location@plt>
    3d8c:	b9400000 	ldr	w0, [x0]
    3d90:	97fff54c 	bl	12c0 <strerror@plt>
    3d94:	aa0003e4 	mov	x4, x0
    3d98:	52800703 	mov	w3, #0x38                  	// #56
    3d9c:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3da0:	91296002 	add	x2, x0, #0xa58
    3da4:	2a1303e1 	mov	w1, w19
    3da8:	f0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    3dac:	912a0000 	add	x0, x0, #0xa80
    3db0:	94000872 	bl	5f78 <spt_ComplainError>
    3db4:	97fff583 	bl	13c0 <__errno_location@plt>
    3db8:	b9400000 	ldr	w0, [x0]
    3dbc:	11404000 	add	w0, w0, #0x10, lsl #12
    3dc0:	1400000f 	b	3dfc <sptNewMatrix+0x160>
	memset(mtx->values, 0, mtx->cap * mtx->stride * sizeof (sptValue));
    3dc4:	f94017e0 	ldr	x0, [sp, #40]
    3dc8:	f9400803 	ldr	x3, [x0, #16]
    3dcc:	f94017e0 	ldr	x0, [sp, #40]
    3dd0:	b9400801 	ldr	w1, [x0, #8]
    3dd4:	f94017e0 	ldr	x0, [sp, #40]
    3dd8:	b9400c00 	ldr	w0, [x0, #12]
    3ddc:	1b007c20 	mul	w0, w1, w0
    3de0:	2a0003e0 	mov	w0, w0
    3de4:	d37ef400 	lsl	x0, x0, #2
    3de8:	aa0003e2 	mov	x2, x0
    3dec:	52800001 	mov	w1, #0x0                   	// #0
    3df0:	aa0303e0 	mov	x0, x3
    3df4:	97fff51f 	bl	1270 <memset@plt>
	return 0;
    3df8:	52800000 	mov	w0, #0x0                   	// #0
}
    3dfc:	f9400bf3 	ldr	x19, [sp, #16]
    3e00:	a8c37bfd 	ldp	x29, x30, [sp], #48
    3e04:	d65f03c0 	ret

0000000000003e08 <sptRandomizeMatrix>:
 * @param ncols fill the specified number of columns
 *
 * The matrix is filled with uniform distributed pseudorandom number in [0, 1]
 * The random number will have a precision of 31 bits out of 51 bits
 */
int sptRandomizeMatrix(sptMatrix *mtx, bool random) {
    3e08:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    3e0c:	910003fd 	mov	x29, sp
    3e10:	f9000bf3 	str	x19, [sp, #16]
    3e14:	f90017e0 	str	x0, [sp, #40]
    3e18:	39009fe1 	strb	w1, [sp, #39]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3e1c:	b9003fff 	str	wzr, [sp, #60]
    3e20:	14000039 	b	3f04 <sptRandomizeMatrix+0xfc>
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3e24:	b9003bff 	str	wzr, [sp, #56]
    3e28:	1400002f 	b	3ee4 <sptRandomizeMatrix+0xdc>
			if (random) {
    3e2c:	39409fe0 	ldrb	w0, [sp, #39]
    3e30:	7100001f 	cmp	w0, #0x0
    3e34:	540002e0 	b.eq	3e90 <sptRandomizeMatrix+0x88>  // b.none
				srand(time(NULL) + (rand()%100) + i + j);
    3e38:	d2800000 	mov	x0, #0x0                   	// #0
    3e3c:	97fff4f5 	bl	1210 <time@plt>
    3e40:	2a0003f3 	mov	w19, w0
    3e44:	97fff513 	bl	1290 <rand@plt>
    3e48:	5290a3e1 	mov	w1, #0x851f                	// #34079
    3e4c:	72aa3d61 	movk	w1, #0x51eb, lsl #16
    3e50:	9b217c01 	smull	x1, w0, w1
    3e54:	d360fc21 	lsr	x1, x1, #32
    3e58:	13057c22 	asr	w2, w1, #5
    3e5c:	131f7c01 	asr	w1, w0, #31
    3e60:	4b010041 	sub	w1, w2, w1
    3e64:	52800c82 	mov	w2, #0x64                  	// #100
    3e68:	1b027c21 	mul	w1, w1, w2
    3e6c:	4b010001 	sub	w1, w0, w1
    3e70:	2a0103e0 	mov	w0, w1
    3e74:	0b000261 	add	w1, w19, w0
    3e78:	b9403fe0 	ldr	w0, [sp, #60]
    3e7c:	0b000021 	add	w1, w1, w0
    3e80:	b9403be0 	ldr	w0, [sp, #56]
    3e84:	0b000020 	add	w0, w1, w0
    3e88:	97fff542 	bl	1390 <srand@plt>
    3e8c:	14000006 	b	3ea4 <sptRandomizeMatrix+0x9c>
			} else {
				srand(1234 + i + j);
    3e90:	b9403fe1 	ldr	w1, [sp, #60]
    3e94:	b9403be0 	ldr	w0, [sp, #56]
    3e98:	0b000020 	add	w0, w1, w0
    3e9c:	11134800 	add	w0, w0, #0x4d2
    3ea0:	97fff53c 	bl	1390 <srand@plt>
			}
			mtx->values[i * mtx->stride + j] = sptRandomValue();
    3ea4:	f94017e0 	ldr	x0, [sp, #40]
    3ea8:	f9400801 	ldr	x1, [x0, #16]
    3eac:	f94017e0 	ldr	x0, [sp, #40]
    3eb0:	b9400c02 	ldr	w2, [x0, #12]
    3eb4:	b9403fe0 	ldr	w0, [sp, #60]
    3eb8:	1b007c42 	mul	w2, w2, w0
    3ebc:	b9403be0 	ldr	w0, [sp, #56]
    3ec0:	0b000040 	add	w0, w2, w0
    3ec4:	2a0003e0 	mov	w0, w0
    3ec8:	d37ef400 	lsl	x0, x0, #2
    3ecc:	8b000033 	add	x19, x1, x0
    3ed0:	940008e4 	bl	6260 <sptRandomValue>
    3ed4:	bd000260 	str	s0, [x19]
		for(sptIndex j=0; j<mtx->ncols; ++j) {
    3ed8:	b9403be0 	ldr	w0, [sp, #56]
    3edc:	11000400 	add	w0, w0, #0x1
    3ee0:	b9003be0 	str	w0, [sp, #56]
    3ee4:	f94017e0 	ldr	x0, [sp, #40]
    3ee8:	b9400400 	ldr	w0, [x0, #4]
    3eec:	b9403be1 	ldr	w1, [sp, #56]
    3ef0:	6b00003f 	cmp	w1, w0
    3ef4:	54fff9c3 	b.cc	3e2c <sptRandomizeMatrix+0x24>  // b.lo, b.ul, b.last
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3ef8:	b9403fe0 	ldr	w0, [sp, #60]
    3efc:	11000400 	add	w0, w0, #0x1
    3f00:	b9003fe0 	str	w0, [sp, #60]
    3f04:	f94017e0 	ldr	x0, [sp, #40]
    3f08:	b9400000 	ldr	w0, [x0]
    3f0c:	b9403fe1 	ldr	w1, [sp, #60]
    3f10:	6b00003f 	cmp	w1, w0
    3f14:	54fff883 	b.cc	3e24 <sptRandomizeMatrix+0x1c>  // b.lo, b.ul, b.last
		}
	return 0;
    3f18:	52800000 	mov	w0, #0x0                   	// #0
}
    3f1c:	f9400bf3 	ldr	x19, [sp, #16]
    3f20:	a8c47bfd 	ldp	x29, x30, [sp], #64
    3f24:	d65f03c0 	ret

0000000000003f28 <sptConstantMatrix>:
 *
 * @param mtx   a pointer to a valid matrix
 * @param val   a given value constant
 *
 */
int sptConstantMatrix(sptMatrix *mtx, sptValue const val) {
    3f28:	d10083ff 	sub	sp, sp, #0x20
    3f2c:	f90007e0 	str	x0, [sp, #8]
    3f30:	bd0007e0 	str	s0, [sp, #4]
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3f34:	b9001fff 	str	wzr, [sp, #28]
    3f38:	1400001b 	b	3fa4 <sptConstantMatrix+0x7c>
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3f3c:	b9001bff 	str	wzr, [sp, #24]
    3f40:	14000011 	b	3f84 <sptConstantMatrix+0x5c>
			mtx->values[i * mtx->stride + j] = val;
    3f44:	f94007e0 	ldr	x0, [sp, #8]
    3f48:	f9400801 	ldr	x1, [x0, #16]
    3f4c:	f94007e0 	ldr	x0, [sp, #8]
    3f50:	b9400c02 	ldr	w2, [x0, #12]
    3f54:	b9401fe0 	ldr	w0, [sp, #28]
    3f58:	1b007c42 	mul	w2, w2, w0
    3f5c:	b9401be0 	ldr	w0, [sp, #24]
    3f60:	0b000040 	add	w0, w2, w0
    3f64:	2a0003e0 	mov	w0, w0
    3f68:	d37ef400 	lsl	x0, x0, #2
    3f6c:	8b000020 	add	x0, x1, x0
    3f70:	bd4007e0 	ldr	s0, [sp, #4]
    3f74:	bd000000 	str	s0, [x0]
		for(sptIndex j=0; j<mtx->ncols; ++j)
    3f78:	b9401be0 	ldr	w0, [sp, #24]
    3f7c:	11000400 	add	w0, w0, #0x1
    3f80:	b9001be0 	str	w0, [sp, #24]
    3f84:	f94007e0 	ldr	x0, [sp, #8]
    3f88:	b9400400 	ldr	w0, [x0, #4]
    3f8c:	b9401be1 	ldr	w1, [sp, #24]
    3f90:	6b00003f 	cmp	w1, w0
    3f94:	54fffd83 	b.cc	3f44 <sptConstantMatrix+0x1c>  // b.lo, b.ul, b.last
	for(sptIndex i=0; i<mtx->nrows; ++i)
    3f98:	b9401fe0 	ldr	w0, [sp, #28]
    3f9c:	11000400 	add	w0, w0, #0x1
    3fa0:	b9001fe0 	str	w0, [sp, #28]
    3fa4:	f94007e0 	ldr	x0, [sp, #8]
    3fa8:	b9400000 	ldr	w0, [x0]
    3fac:	b9401fe1 	ldr	w1, [sp, #28]
    3fb0:	6b00003f 	cmp	w1, w0
    3fb4:	54fffc43 	b.cc	3f3c <sptConstantMatrix+0x14>  // b.lo, b.ul, b.last
	return 0;
    3fb8:	52800000 	mov	w0, #0x0                   	// #0
}
    3fbc:	910083ff 	add	sp, sp, #0x20
    3fc0:	d65f03c0 	ret

0000000000003fc4 <sptFreeMatrix>:
 * @param mtx a pointer to a valid matrix
 *
 * By using `sptFreeMatrix`, a valid matrix would become uninitialized and
 * should not be used anymore prior to another initialization
 */
void sptFreeMatrix(sptMatrix *mtx) {
    3fc4:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    3fc8:	910003fd 	mov	x29, sp
    3fcc:	f9000fe0 	str	x0, [sp, #24]
	free(mtx->values);
    3fd0:	f9400fe0 	ldr	x0, [sp, #24]
    3fd4:	f9400800 	ldr	x0, [x0, #16]
    3fd8:	97fff4de 	bl	1350 <free@plt>
	mtx->nrows = 0;
    3fdc:	f9400fe0 	ldr	x0, [sp, #24]
    3fe0:	b900001f 	str	wzr, [x0]
	mtx->ncols = 0;
    3fe4:	f9400fe0 	ldr	x0, [sp, #24]
    3fe8:	b900041f 	str	wzr, [x0, #4]
	mtx->cap = 0;
    3fec:	f9400fe0 	ldr	x0, [sp, #24]
    3ff0:	b900081f 	str	wzr, [x0, #8]
	mtx->stride = 0;
    3ff4:	f9400fe0 	ldr	x0, [sp, #24]
    3ff8:	b9000c1f 	str	wzr, [x0, #12]
}
    3ffc:	d503201f 	nop
    4000:	a8c27bfd 	ldp	x29, x30, [sp], #32
    4004:	d65f03c0 	ret

0000000000004008 <sptSparseTensorDensity>:
#include <bits/types/FILE.h>
#include <stdio.h>


double sptSparseTensorDensity(sptSparseTensor const * const tsr)
{
    4008:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    400c:	910003fd 	mov	x29, sp
    4010:	f9000fe0 	str	x0, [sp, #24]
	double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
    4014:	f9400fe0 	ldr	x0, [sp, #24]
    4018:	fd400c00 	ldr	d0, [x0, #24]
    401c:	7e61d802 	ucvtf	d2, d0
    4020:	f9400fe0 	ldr	x0, [sp, #24]
    4024:	b9400000 	ldr	w0, [x0]
    4028:	1e630000 	ucvtf	d0, w0
    402c:	1e6e1001 	fmov	d1, #1.000000000000000000e+00
    4030:	1e601820 	fdiv	d0, d1, d0
    4034:	1e604001 	fmov	d1, d0
    4038:	1e604040 	fmov	d0, d2
    403c:	97fff465 	bl	11d0 <pow@plt>
    4040:	fd0017e0 	str	d0, [sp, #40]
	double density = 1.0;
    4044:	1e6e1000 	fmov	d0, #1.000000000000000000e+00
    4048:	fd001fe0 	str	d0, [sp, #56]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    404c:	b90037ff 	str	wzr, [sp, #52]
    4050:	14000010 	b	4090 <sptSparseTensorDensity+0x88>
		density *= root / (double)tsr->ndims[m];
    4054:	f9400fe0 	ldr	x0, [sp, #24]
    4058:	f9400801 	ldr	x1, [x0, #16]
    405c:	b94037e0 	ldr	w0, [sp, #52]
    4060:	d37ef400 	lsl	x0, x0, #2
    4064:	8b000020 	add	x0, x1, x0
    4068:	b9400000 	ldr	w0, [x0]
    406c:	1e630000 	ucvtf	d0, w0
    4070:	fd4017e1 	ldr	d1, [sp, #40]
    4074:	1e601820 	fdiv	d0, d1, d0
    4078:	fd401fe1 	ldr	d1, [sp, #56]
    407c:	1e600820 	fmul	d0, d1, d0
    4080:	fd001fe0 	str	d0, [sp, #56]
	for(sptIndex m=0; m < tsr->nmodes; ++m) {
    4084:	b94037e0 	ldr	w0, [sp, #52]
    4088:	11000400 	add	w0, w0, #0x1
    408c:	b90037e0 	str	w0, [sp, #52]
    4090:	f9400fe0 	ldr	x0, [sp, #24]
    4094:	b9400000 	ldr	w0, [x0]
    4098:	b94037e1 	ldr	w1, [sp, #52]
    409c:	6b00003f 	cmp	w1, w0
    40a0:	54fffda3 	b.cc	4054 <sptSparseTensorDensity+0x4c>  // b.lo, b.ul, b.last
	}

	return density;
    40a4:	fd401fe0 	ldr	d0, [sp, #56]
}
    40a8:	a8c47bfd 	ldp	x29, x30, [sp], #64
    40ac:	d65f03c0 	ret

00000000000040b0 <sptSparseTensorStatus>:



void sptSparseTensorStatus(sptSparseTensor *tsr, FILE *fp)
{
    40b0:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    40b4:	910003fd 	mov	x29, sp
    40b8:	f9000fe0 	str	x0, [sp, #24]
    40bc:	f9000be1 	str	x1, [sp, #16]
	fprintf(fp, "COO Sparse Tensor information (use sptIndex, sptValue))---------\n");
    40c0:	f9400be3 	ldr	x3, [sp, #16]
    40c4:	d2800822 	mov	x2, #0x41                  	// #65
    40c8:	d2800021 	mov	x1, #0x1                   	// #1
    40cc:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    40d0:	912a2000 	add	x0, x0, #0xa88
    40d4:	97fff4a3 	bl	1360 <fwrite@plt>
	fprintf(fp, "DIMS = %"PASTA_PRI_INDEX, tsr->ndims[0]);
    40d8:	f9400fe0 	ldr	x0, [sp, #24]
    40dc:	f9400800 	ldr	x0, [x0, #16]
    40e0:	b9400000 	ldr	w0, [x0]
    40e4:	2a0003e2 	mov	w2, w0
    40e8:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    40ec:	912b4001 	add	x1, x0, #0xad0
    40f0:	f9400be0 	ldr	x0, [sp, #16]
    40f4:	97fff4bf 	bl	13f0 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    40f8:	52800020 	mov	w0, #0x1                   	// #1
    40fc:	b9002fe0 	str	w0, [sp, #44]
    4100:	1400000f 	b	413c <sptSparseTensorStatus+0x8c>
		fprintf(fp, "x%"PASTA_PRI_INDEX, tsr->ndims[m]);
    4104:	f9400fe0 	ldr	x0, [sp, #24]
    4108:	f9400801 	ldr	x1, [x0, #16]
    410c:	b9402fe0 	ldr	w0, [sp, #44]
    4110:	d37ef400 	lsl	x0, x0, #2
    4114:	8b000020 	add	x0, x1, x0
    4118:	b9400000 	ldr	w0, [x0]
    411c:	2a0003e2 	mov	w2, w0
    4120:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4124:	912b8001 	add	x1, x0, #0xae0
    4128:	f9400be0 	ldr	x0, [sp, #16]
    412c:	97fff4b1 	bl	13f0 <fprintf@plt>
	for(sptIndex m=1; m < tsr->nmodes; ++m) {
    4130:	b9402fe0 	ldr	w0, [sp, #44]
    4134:	11000400 	add	w0, w0, #0x1
    4138:	b9002fe0 	str	w0, [sp, #44]
    413c:	f9400fe0 	ldr	x0, [sp, #24]
    4140:	b9400000 	ldr	w0, [x0]
    4144:	b9402fe1 	ldr	w1, [sp, #44]
    4148:	6b00003f 	cmp	w1, w0
    414c:	54fffdc3 	b.cc	4104 <sptSparseTensorStatus+0x54>  // b.lo, b.ul, b.last
	}
	fprintf(fp, " NNZ = %"PASTA_PRI_NNZ_INDEX, tsr->nnz);
    4150:	f9400fe0 	ldr	x0, [sp, #24]
    4154:	f9400c00 	ldr	x0, [x0, #24]
    4158:	aa0003e2 	mov	x2, x0
    415c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4160:	912ba001 	add	x1, x0, #0xae8
    4164:	f9400be0 	ldr	x0, [sp, #16]
    4168:	97fff4a2 	bl	13f0 <fprintf@plt>
	fprintf(fp, " DENSITY = %e\n" , sptSparseTensorDensity(tsr));
    416c:	f9400fe0 	ldr	x0, [sp, #24]
    4170:	97ffffa6 	bl	4008 <sptSparseTensorDensity>
    4174:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4178:	912be001 	add	x1, x0, #0xaf8
    417c:	f9400be0 	ldr	x0, [sp, #16]
    4180:	97fff49c 	bl	13f0 <fprintf@plt>

	fprintf(fp, "Average slice length (c): ");
    4184:	f9400be3 	ldr	x3, [sp, #16]
    4188:	d2800342 	mov	x2, #0x1a                  	// #26
    418c:	d2800021 	mov	x1, #0x1                   	// #1
    4190:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4194:	912c2000 	add	x0, x0, #0xb08
    4198:	97fff472 	bl	1360 <fwrite@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    419c:	b9002bff 	str	wzr, [sp, #40]
    41a0:	14000013 	b	41ec <sptSparseTensorStatus+0x13c>
		fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
    41a4:	f9400fe0 	ldr	x0, [sp, #24]
    41a8:	fd400c00 	ldr	d0, [x0, #24]
    41ac:	7e61d801 	ucvtf	d1, d0
    41b0:	f9400fe0 	ldr	x0, [sp, #24]
    41b4:	f9400801 	ldr	x1, [x0, #16]
    41b8:	b9402be0 	ldr	w0, [sp, #40]
    41bc:	d37ef400 	lsl	x0, x0, #2
    41c0:	8b000020 	add	x0, x1, x0
    41c4:	b9400000 	ldr	w0, [x0]
    41c8:	1e630000 	ucvtf	d0, w0
    41cc:	1e601820 	fdiv	d0, d1, d0
    41d0:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    41d4:	912ca001 	add	x1, x0, #0xb28
    41d8:	f9400be0 	ldr	x0, [sp, #16]
    41dc:	97fff485 	bl	13f0 <fprintf@plt>
	for(sptIndex m=0; m < tsr->nmodes - 1; ++m) {
    41e0:	b9402be0 	ldr	w0, [sp, #40]
    41e4:	11000400 	add	w0, w0, #0x1
    41e8:	b9002be0 	str	w0, [sp, #40]
    41ec:	f9400fe0 	ldr	x0, [sp, #24]
    41f0:	b9400000 	ldr	w0, [x0]
    41f4:	51000400 	sub	w0, w0, #0x1
    41f8:	b9402be1 	ldr	w1, [sp, #40]
    41fc:	6b00003f 	cmp	w1, w0
    4200:	54fffd23 	b.cc	41a4 <sptSparseTensorStatus+0xf4>  // b.lo, b.ul, b.last
	}
	fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);
    4204:	f9400fe0 	ldr	x0, [sp, #24]
    4208:	fd400c00 	ldr	d0, [x0, #24]
    420c:	7e61d801 	ucvtf	d1, d0
    4210:	f9400fe0 	ldr	x0, [sp, #24]
    4214:	f9400801 	ldr	x1, [x0, #16]
    4218:	f9400fe0 	ldr	x0, [sp, #24]
    421c:	b9400000 	ldr	w0, [x0]
    4220:	51000400 	sub	w0, w0, #0x1
    4224:	2a0003e0 	mov	w0, w0
    4228:	d37ef400 	lsl	x0, x0, #2
    422c:	8b000020 	add	x0, x1, x0
    4230:	b9400000 	ldr	w0, [x0]
    4234:	1e630000 	ucvtf	d0, w0
    4238:	1e601820 	fdiv	d0, d1, d0
    423c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4240:	912ce001 	add	x1, x0, #0xb38
    4244:	f9400be0 	ldr	x0, [sp, #16]
    4248:	97fff46a 	bl	13f0 <fprintf@plt>

	char * bytestr = sptBytesString(tsr->nnz * (sizeof(sptIndex) * tsr->nmodes + sizeof(sptValue)));
    424c:	f9400fe0 	ldr	x0, [sp, #24]
    4250:	b9400000 	ldr	w0, [x0]
    4254:	2a0003e0 	mov	w0, w0
    4258:	91000401 	add	x1, x0, #0x1
    425c:	f9400fe0 	ldr	x0, [sp, #24]
    4260:	f9400c00 	ldr	x0, [x0, #24]
    4264:	9b007c20 	mul	x0, x1, x0
    4268:	d37ef400 	lsl	x0, x0, #2
    426c:	940007c1 	bl	6170 <sptBytesString>
    4270:	f90013e0 	str	x0, [sp, #32]
	fprintf(fp, "COO-STORAGE = %s\n", bytestr);
    4274:	f94013e2 	ldr	x2, [sp, #32]
    4278:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    427c:	912d0001 	add	x1, x0, #0xb40
    4280:	f9400be0 	ldr	x0, [sp, #16]
    4284:	97fff45b 	bl	13f0 <fprintf@plt>
	fprintf(fp, "\n");
    4288:	f9400be1 	ldr	x1, [sp, #16]
    428c:	52800140 	mov	w0, #0xa                   	// #10
    4290:	97fff3c4 	bl	11a0 <fputc@plt>
	free(bytestr);
    4294:	f94013e0 	ldr	x0, [sp, #32]
    4298:	97fff42e 	bl	1350 <free@plt>
}
    429c:	d503201f 	nop
    42a0:	a8c37bfd 	ldp	x29, x30, [sp], #48
    42a4:	d65f03c0 	ret

00000000000042a8 <sptMTTKRP>:
 * products of dense factor matrices, the output is the updated dense matrix for the "mode".
 */
int sptMTTKRP(sptSparseTensor const * const X,
							sptMatrix * mats[],     // mats[nmodes] as temporary space.
							sptIndex const mats_order[],    // Correspond to the mode order of X.
							sptIndex const mode) {
    42a8:	a9ae7bfd 	stp	x29, x30, [sp, #-288]!
    42ac:	910003fd 	mov	x29, sp
    42b0:	f90017e0 	str	x0, [sp, #40]
    42b4:	f90013e1 	str	x1, [sp, #32]
    42b8:	f9000fe2 	str	x2, [sp, #24]
    42bc:	b90017e3 	str	w3, [sp, #20]

	sptIndex const nmodes = X->nmodes;
    42c0:	f94017e0 	ldr	x0, [sp, #40]
    42c4:	b9400000 	ldr	w0, [x0]
    42c8:	b900ffe0 	str	w0, [sp, #252]
	if(nmodes == 3) {
    42cc:	b940ffe0 	ldr	w0, [sp, #252]
    42d0:	71000c1f 	cmp	w0, #0x3
    42d4:	54000181 	b.ne	4304 <sptMTTKRP+0x5c>  // b.any
		sptAssert(sptMTTKRP_3D(X, mats, mats_order, mode) == 0);
    42d8:	b94017e3 	ldr	w3, [sp, #20]
    42dc:	f9400fe2 	ldr	x2, [sp, #24]
    42e0:	f94013e1 	ldr	x1, [sp, #32]
    42e4:	f94017e0 	ldr	x0, [sp, #40]
    42e8:	94000158 	bl	4848 <sptMTTKRP_3D>
    42ec:	7100001f 	cmp	w0, #0x0
    42f0:	54000060 	b.eq	42fc <sptMTTKRP+0x54>  // b.none
    42f4:	12800000 	mov	w0, #0xffffffff            	// #-1
    42f8:	97fff3a2 	bl	1180 <exit@plt>
		return 0;
    42fc:	52800000 	mov	w0, #0x0                   	// #0
    4300:	14000150 	b	4840 <sptMTTKRP+0x598>
	}

	sptNnzIndex const nnz = X->nnz;
    4304:	f94017e0 	ldr	x0, [sp, #40]
    4308:	f9400c00 	ldr	x0, [x0, #24]
    430c:	f9007be0 	str	x0, [sp, #240]
	sptIndex const * const ndims = X->ndims;
    4310:	f94017e0 	ldr	x0, [sp, #40]
    4314:	f9400800 	ldr	x0, [x0, #16]
    4318:	f90077e0 	str	x0, [sp, #232]
	sptValue const * const restrict vals = X->values.data;
    431c:	f94017e0 	ldr	x0, [sp, #40]
    4320:	f9401c00 	ldr	x0, [x0, #56]
    4324:	f90073e0 	str	x0, [sp, #224]
	sptIndex const stride = mats[0]->stride;
    4328:	f94013e0 	ldr	x0, [sp, #32]
    432c:	f9400000 	ldr	x0, [x0]
    4330:	b9400c00 	ldr	w0, [x0, #12]
    4334:	b900dfe0 	str	w0, [sp, #220]
	sptValueVector scratch;  // Temporary array

	/* Check the mats. */
	for(sptIndex i=0; i<nmodes; ++i) {
    4338:	b9011fff 	str	wzr, [sp, #284]
    433c:	14000035 	b	4410 <sptMTTKRP+0x168>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4340:	b9411fe0 	ldr	w0, [sp, #284]
    4344:	d37df000 	lsl	x0, x0, #3
    4348:	f94013e1 	ldr	x1, [sp, #32]
    434c:	8b000020 	add	x0, x1, x0
    4350:	f9400000 	ldr	x0, [x0]
    4354:	b9400401 	ldr	w1, [x0, #4]
    4358:	b940ffe0 	ldr	w0, [sp, #252]
    435c:	d37df000 	lsl	x0, x0, #3
    4360:	f94013e2 	ldr	x2, [sp, #32]
    4364:	8b000040 	add	x0, x2, x0
    4368:	f9400000 	ldr	x0, [x0]
    436c:	b9400400 	ldr	w0, [x0, #4]
    4370:	6b00003f 	cmp	w1, w0
    4374:	54000180 	b.eq	43a4 <sptMTTKRP+0xfc>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    4378:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    437c:	912d6004 	add	x4, x0, #0xb58
    4380:	528007c3 	mov	w3, #0x3e                  	// #62
    4384:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4388:	912e0002 	add	x2, x0, #0xb80
    438c:	52800041 	mov	w1, #0x2                   	// #2
    4390:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4394:	912ea000 	add	x0, x0, #0xba8
    4398:	940006f8 	bl	5f78 <spt_ComplainError>
    439c:	52800040 	mov	w0, #0x2                   	// #2
    43a0:	14000128 	b	4840 <sptMTTKRP+0x598>
		}
		if(mats[i]->nrows != ndims[i]) {
    43a4:	b9411fe0 	ldr	w0, [sp, #284]
    43a8:	d37df000 	lsl	x0, x0, #3
    43ac:	f94013e1 	ldr	x1, [sp, #32]
    43b0:	8b000020 	add	x0, x1, x0
    43b4:	f9400000 	ldr	x0, [x0]
    43b8:	b9400001 	ldr	w1, [x0]
    43bc:	b9411fe0 	ldr	w0, [sp, #284]
    43c0:	d37ef400 	lsl	x0, x0, #2
    43c4:	f94077e2 	ldr	x2, [sp, #232]
    43c8:	8b000040 	add	x0, x2, x0
    43cc:	b9400000 	ldr	w0, [x0]
    43d0:	6b00003f 	cmp	w1, w0
    43d4:	54000180 	b.eq	4404 <sptMTTKRP+0x15c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    43d8:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    43dc:	912f0004 	add	x4, x0, #0xbc0
    43e0:	52800823 	mov	w3, #0x41                  	// #65
    43e4:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    43e8:	912e0002 	add	x2, x0, #0xb80
    43ec:	52800041 	mov	w1, #0x2                   	// #2
    43f0:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    43f4:	912ea000 	add	x0, x0, #0xba8
    43f8:	940006e0 	bl	5f78 <spt_ComplainError>
    43fc:	52800040 	mov	w0, #0x2                   	// #2
    4400:	14000110 	b	4840 <sptMTTKRP+0x598>
	for(sptIndex i=0; i<nmodes; ++i) {
    4404:	b9411fe0 	ldr	w0, [sp, #284]
    4408:	11000400 	add	w0, w0, #0x1
    440c:	b9011fe0 	str	w0, [sp, #284]
    4410:	b9411fe1 	ldr	w1, [sp, #284]
    4414:	b940ffe0 	ldr	w0, [sp, #252]
    4418:	6b00003f 	cmp	w1, w0
    441c:	54fff923 	b.cc	4340 <sptMTTKRP+0x98>  // b.lo, b.ul, b.last
		}
	}

	sptIndex const tmpI = mats[mode]->nrows;
    4420:	b94017e0 	ldr	w0, [sp, #20]
    4424:	d37df000 	lsl	x0, x0, #3
    4428:	f94013e1 	ldr	x1, [sp, #32]
    442c:	8b000020 	add	x0, x1, x0
    4430:	f9400000 	ldr	x0, [x0]
    4434:	b9400000 	ldr	w0, [x0]
    4438:	b900dbe0 	str	w0, [sp, #216]
	sptIndex const R = mats[mode]->ncols;
    443c:	b94017e0 	ldr	w0, [sp, #20]
    4440:	d37df000 	lsl	x0, x0, #3
    4444:	f94013e1 	ldr	x1, [sp, #32]
    4448:	8b000020 	add	x0, x1, x0
    444c:	f9400000 	ldr	x0, [x0]
    4450:	b9400400 	ldr	w0, [x0, #4]
    4454:	b900d7e0 	str	w0, [sp, #212]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    4458:	f94017e0 	ldr	x0, [sp, #40]
    445c:	f9401002 	ldr	x2, [x0, #32]
    4460:	b94017e1 	ldr	w1, [sp, #20]
    4464:	aa0103e0 	mov	x0, x1
    4468:	d37ff800 	lsl	x0, x0, #1
    446c:	8b010000 	add	x0, x0, x1
    4470:	d37df000 	lsl	x0, x0, #3
    4474:	8b000040 	add	x0, x2, x0
    4478:	f9400800 	ldr	x0, [x0, #16]
    447c:	f90067e0 	str	x0, [sp, #200]
	sptMatrix * const restrict M = mats[nmodes];
    4480:	b940ffe0 	ldr	w0, [sp, #252]
    4484:	d37df000 	lsl	x0, x0, #3
    4488:	f94013e1 	ldr	x1, [sp, #32]
    448c:	8b000020 	add	x0, x1, x0
    4490:	f9400000 	ldr	x0, [x0]
    4494:	f90063e0 	str	x0, [sp, #192]
	sptValue * const restrict mvals = M->values;
    4498:	f94063e0 	ldr	x0, [sp, #192]
    449c:	f9400800 	ldr	x0, [x0, #16]
    44a0:	f9005fe0 	str	x0, [sp, #184]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    44a4:	b940dbe1 	ldr	w1, [sp, #216]
    44a8:	b940dfe0 	ldr	w0, [sp, #220]
    44ac:	1b007c20 	mul	w0, w1, w0
    44b0:	2a0003e0 	mov	w0, w0
    44b4:	d37ef400 	lsl	x0, x0, #2
    44b8:	aa0003e2 	mov	x2, x0
    44bc:	52800001 	mov	w1, #0x0                   	// #0
    44c0:	f9405fe0 	ldr	x0, [sp, #184]
    44c4:	97fff36b 	bl	1270 <memset@plt>
	sptNewValueVector(&scratch, R, R);
    44c8:	b940d7e1 	ldr	w1, [sp, #212]
    44cc:	b940d7e2 	ldr	w2, [sp, #212]
    44d0:	910103e0 	add	x0, sp, #0x40
    44d4:	97fff76c 	bl	2284 <sptNewValueVector>
	sptConstantValueVector(&scratch, 0);
    44d8:	910103e0 	add	x0, sp, #0x40
    44dc:	0f000400 	movi	v0.2s, #0x0
    44e0:	97fff7b4 	bl	23b0 <sptConstantValueVector>


	sptTimer timer;
	sptNewTimer(&timer, 0);
    44e4:	9100e3e0 	add	x0, sp, #0x38
    44e8:	52800001 	mov	w1, #0x0                   	// #0
    44ec:	9400051c 	bl	595c <sptNewTimer>
	double comp_time, total_time;

	/* Computation */
	sptStartTimer(timer);
    44f0:	f9401fe0 	ldr	x0, [sp, #56]
    44f4:	9400053a 	bl	59dc <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    44f8:	f9008bff 	str	xzr, [sp, #272]
    44fc:	140000b8 	b	47dc <sptMTTKRP+0x534>

		sptIndex times_mat_index = mats_order[1];
    4500:	f9400fe0 	ldr	x0, [sp, #24]
    4504:	b9400400 	ldr	w0, [x0, #4]
    4508:	b900a7e0 	str	w0, [sp, #164]
		sptMatrix * times_mat = mats[times_mat_index];
    450c:	b940a7e0 	ldr	w0, [sp, #164]
    4510:	d37df000 	lsl	x0, x0, #3
    4514:	f94013e1 	ldr	x1, [sp, #32]
    4518:	8b000020 	add	x0, x1, x0
    451c:	f9400000 	ldr	x0, [x0]
    4520:	f9004fe0 	str	x0, [sp, #152]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    4524:	f94017e0 	ldr	x0, [sp, #40]
    4528:	f9401002 	ldr	x2, [x0, #32]
    452c:	b940a7e1 	ldr	w1, [sp, #164]
    4530:	aa0103e0 	mov	x0, x1
    4534:	d37ff800 	lsl	x0, x0, #1
    4538:	8b010000 	add	x0, x0, x1
    453c:	d37df000 	lsl	x0, x0, #3
    4540:	8b000040 	add	x0, x2, x0
    4544:	f9400800 	ldr	x0, [x0, #16]
    4548:	f9004be0 	str	x0, [sp, #144]
		sptIndex tmp_i = times_inds[x];
    454c:	f9408be0 	ldr	x0, [sp, #272]
    4550:	d37ef400 	lsl	x0, x0, #2
    4554:	f9404be1 	ldr	x1, [sp, #144]
    4558:	8b000020 	add	x0, x1, x0
    455c:	b9400000 	ldr	w0, [x0]
    4560:	b9008fe0 	str	w0, [sp, #140]
		sptValue const entry = vals[x];
    4564:	f9408be0 	ldr	x0, [sp, #272]
    4568:	d37ef400 	lsl	x0, x0, #2
    456c:	f94073e1 	ldr	x1, [sp, #224]
    4570:	8b000020 	add	x0, x1, x0
    4574:	bd400000 	ldr	s0, [x0]
    4578:	bd008be0 	str	s0, [sp, #136]
		sptIndex tmp_mult = tmp_i * stride;
    457c:	b9408fe1 	ldr	w1, [sp, #140]
    4580:	b940dfe0 	ldr	w0, [sp, #220]
    4584:	1b007c20 	mul	w0, w1, w0
    4588:	b90087e0 	str	w0, [sp, #132]
		sptValue* times_mat_values = times_mat->values + tmp_mult;
    458c:	f9404fe0 	ldr	x0, [sp, #152]
    4590:	f9400801 	ldr	x1, [x0, #16]
    4594:	b94087e0 	ldr	w0, [sp, #132]
    4598:	d37ef400 	lsl	x0, x0, #2
    459c:	8b000020 	add	x0, x1, x0
    45a0:	f9003fe0 	str	x0, [sp, #120]
		for(sptIndex r=0; r<R; ++r) {
    45a4:	b9010fff 	str	wzr, [sp, #268]
    45a8:	14000010 	b	45e8 <sptMTTKRP+0x340>
			scratch.data[r] = entry * times_mat_values[r];
    45ac:	b9410fe0 	ldr	w0, [sp, #268]
    45b0:	d37ef400 	lsl	x0, x0, #2
    45b4:	f9403fe1 	ldr	x1, [sp, #120]
    45b8:	8b000020 	add	x0, x1, x0
    45bc:	bd400001 	ldr	s1, [x0]
    45c0:	f9402be1 	ldr	x1, [sp, #80]
    45c4:	b9410fe0 	ldr	w0, [sp, #268]
    45c8:	d37ef400 	lsl	x0, x0, #2
    45cc:	8b000020 	add	x0, x1, x0
    45d0:	bd408be0 	ldr	s0, [sp, #136]
    45d4:	1e200820 	fmul	s0, s1, s0
    45d8:	bd000000 	str	s0, [x0]
		for(sptIndex r=0; r<R; ++r) {
    45dc:	b9410fe0 	ldr	w0, [sp, #268]
    45e0:	11000400 	add	w0, w0, #0x1
    45e4:	b9010fe0 	str	w0, [sp, #268]
    45e8:	b9410fe1 	ldr	w1, [sp, #268]
    45ec:	b940d7e0 	ldr	w0, [sp, #212]
    45f0:	6b00003f 	cmp	w1, w0
    45f4:	54fffdc3 	b.cc	45ac <sptMTTKRP+0x304>  // b.lo, b.ul, b.last
		}

		for(sptIndex i=2; i<nmodes; ++i) {
    45f8:	52800040 	mov	w0, #0x2                   	// #2
    45fc:	b9010be0 	str	w0, [sp, #264]
    4600:	14000045 	b	4714 <sptMTTKRP+0x46c>
			times_mat_index = mats_order[i];
    4604:	b9410be0 	ldr	w0, [sp, #264]
    4608:	d37ef400 	lsl	x0, x0, #2
    460c:	f9400fe1 	ldr	x1, [sp, #24]
    4610:	8b000020 	add	x0, x1, x0
    4614:	b9400000 	ldr	w0, [x0]
    4618:	b900a7e0 	str	w0, [sp, #164]
			times_mat = mats[times_mat_index];
    461c:	b940a7e0 	ldr	w0, [sp, #164]
    4620:	d37df000 	lsl	x0, x0, #3
    4624:	f94013e1 	ldr	x1, [sp, #32]
    4628:	8b000020 	add	x0, x1, x0
    462c:	f9400000 	ldr	x0, [x0]
    4630:	f9004fe0 	str	x0, [sp, #152]
			times_inds = X->inds[times_mat_index].data;
    4634:	f94017e0 	ldr	x0, [sp, #40]
    4638:	f9401002 	ldr	x2, [x0, #32]
    463c:	b940a7e1 	ldr	w1, [sp, #164]
    4640:	aa0103e0 	mov	x0, x1
    4644:	d37ff800 	lsl	x0, x0, #1
    4648:	8b010000 	add	x0, x0, x1
    464c:	d37df000 	lsl	x0, x0, #3
    4650:	8b000040 	add	x0, x2, x0
    4654:	f9400800 	ldr	x0, [x0, #16]
    4658:	f9004be0 	str	x0, [sp, #144]
			tmp_i = times_inds[x];
    465c:	f9408be0 	ldr	x0, [sp, #272]
    4660:	d37ef400 	lsl	x0, x0, #2
    4664:	f9404be1 	ldr	x1, [sp, #144]
    4668:	8b000020 	add	x0, x1, x0
    466c:	b9400000 	ldr	w0, [x0]
    4670:	b9008fe0 	str	w0, [sp, #140]
			tmp_mult = tmp_i * stride;
    4674:	b9408fe1 	ldr	w1, [sp, #140]
    4678:	b940dfe0 	ldr	w0, [sp, #220]
    467c:	1b007c20 	mul	w0, w1, w0
    4680:	b90087e0 	str	w0, [sp, #132]
			sptValue* times_mat_values = times_mat->values + tmp_mult;
    4684:	f9404fe0 	ldr	x0, [sp, #152]
    4688:	f9400801 	ldr	x1, [x0, #16]
    468c:	b94087e0 	ldr	w0, [sp, #132]
    4690:	d37ef400 	lsl	x0, x0, #2
    4694:	8b000020 	add	x0, x1, x0
    4698:	f90033e0 	str	x0, [sp, #96]
			sptValue* scratch_data = scratch.data;
    469c:	f9402be0 	ldr	x0, [sp, #80]
    46a0:	f9002fe0 	str	x0, [sp, #88]
			for(sptIndex r=0; r<R; ++r) {
    46a4:	b90107ff 	str	wzr, [sp, #260]
    46a8:	14000014 	b	46f8 <sptMTTKRP+0x450>
				scratch_data[r] *= times_mat_values[r];
    46ac:	b94107e0 	ldr	w0, [sp, #260]
    46b0:	d37ef400 	lsl	x0, x0, #2
    46b4:	f9402fe1 	ldr	x1, [sp, #88]
    46b8:	8b000020 	add	x0, x1, x0
    46bc:	bd400001 	ldr	s1, [x0]
    46c0:	b94107e0 	ldr	w0, [sp, #260]
    46c4:	d37ef400 	lsl	x0, x0, #2
    46c8:	f94033e1 	ldr	x1, [sp, #96]
    46cc:	8b000020 	add	x0, x1, x0
    46d0:	bd400000 	ldr	s0, [x0]
    46d4:	b94107e0 	ldr	w0, [sp, #260]
    46d8:	d37ef400 	lsl	x0, x0, #2
    46dc:	f9402fe1 	ldr	x1, [sp, #88]
    46e0:	8b000020 	add	x0, x1, x0
    46e4:	1e200820 	fmul	s0, s1, s0
    46e8:	bd000000 	str	s0, [x0]
			for(sptIndex r=0; r<R; ++r) {
    46ec:	b94107e0 	ldr	w0, [sp, #260]
    46f0:	11000400 	add	w0, w0, #0x1
    46f4:	b90107e0 	str	w0, [sp, #260]
    46f8:	b94107e1 	ldr	w1, [sp, #260]
    46fc:	b940d7e0 	ldr	w0, [sp, #212]
    4700:	6b00003f 	cmp	w1, w0
    4704:	54fffd43 	b.cc	46ac <sptMTTKRP+0x404>  // b.lo, b.ul, b.last
		for(sptIndex i=2; i<nmodes; ++i) {
    4708:	b9410be0 	ldr	w0, [sp, #264]
    470c:	11000400 	add	w0, w0, #0x1
    4710:	b9010be0 	str	w0, [sp, #264]
    4714:	b9410be1 	ldr	w1, [sp, #264]
    4718:	b940ffe0 	ldr	w0, [sp, #252]
    471c:	6b00003f 	cmp	w1, w0
    4720:	54fff723 	b.cc	4604 <sptMTTKRP+0x35c>  // b.lo, b.ul, b.last
			}
		}

		sptIndex const mode_i = mode_ind[x];
    4724:	f9408be0 	ldr	x0, [sp, #272]
    4728:	d37ef400 	lsl	x0, x0, #2
    472c:	f94067e1 	ldr	x1, [sp, #200]
    4730:	8b000020 	add	x0, x1, x0
    4734:	b9400000 	ldr	w0, [x0]
    4738:	b90077e0 	str	w0, [sp, #116]
		sptIndex tmp_mode = mode_i * stride;
    473c:	b94077e1 	ldr	w1, [sp, #116]
    4740:	b940dfe0 	ldr	w0, [sp, #220]
    4744:	1b007c20 	mul	w0, w1, w0
    4748:	b90073e0 	str	w0, [sp, #112]
		sptValue* scratch_data = scratch.data;
    474c:	f9402be0 	ldr	x0, [sp, #80]
    4750:	f90037e0 	str	x0, [sp, #104]
		for(sptIndex r=0; r<R; ++r) {
    4754:	b90103ff 	str	wzr, [sp, #256]
    4758:	1400001a 	b	47c0 <sptMTTKRP+0x518>
			mvals[tmp_mode + r] += scratch_data[r];
    475c:	b94073e1 	ldr	w1, [sp, #112]
    4760:	b94103e0 	ldr	w0, [sp, #256]
    4764:	0b000020 	add	w0, w1, w0
    4768:	2a0003e0 	mov	w0, w0
    476c:	d37ef400 	lsl	x0, x0, #2
    4770:	f9405fe1 	ldr	x1, [sp, #184]
    4774:	8b000020 	add	x0, x1, x0
    4778:	bd400001 	ldr	s1, [x0]
    477c:	b94103e0 	ldr	w0, [sp, #256]
    4780:	d37ef400 	lsl	x0, x0, #2
    4784:	f94037e1 	ldr	x1, [sp, #104]
    4788:	8b000020 	add	x0, x1, x0
    478c:	bd400000 	ldr	s0, [x0]
    4790:	b94073e1 	ldr	w1, [sp, #112]
    4794:	b94103e0 	ldr	w0, [sp, #256]
    4798:	0b000020 	add	w0, w1, w0
    479c:	2a0003e0 	mov	w0, w0
    47a0:	d37ef400 	lsl	x0, x0, #2
    47a4:	f9405fe1 	ldr	x1, [sp, #184]
    47a8:	8b000020 	add	x0, x1, x0
    47ac:	1e202820 	fadd	s0, s1, s0
    47b0:	bd000000 	str	s0, [x0]
		for(sptIndex r=0; r<R; ++r) {
    47b4:	b94103e0 	ldr	w0, [sp, #256]
    47b8:	11000400 	add	w0, w0, #0x1
    47bc:	b90103e0 	str	w0, [sp, #256]
    47c0:	b94103e1 	ldr	w1, [sp, #256]
    47c4:	b940d7e0 	ldr	w0, [sp, #212]
    47c8:	6b00003f 	cmp	w1, w0
    47cc:	54fffc83 	b.cc	475c <sptMTTKRP+0x4b4>  // b.lo, b.ul, b.last
	for(sptNnzIndex x=0; x<nnz; ++x) {
    47d0:	f9408be0 	ldr	x0, [sp, #272]
    47d4:	91000400 	add	x0, x0, #0x1
    47d8:	f9008be0 	str	x0, [sp, #272]
    47dc:	f9408be1 	ldr	x1, [sp, #272]
    47e0:	f9407be0 	ldr	x0, [sp, #240]
    47e4:	eb00003f 	cmp	x1, x0
    47e8:	54ffe8c3 	b.cc	4500 <sptMTTKRP+0x258>  // b.lo, b.ul, b.last
		}
	}
	sptStopTimer(timer);
    47ec:	f9401fe0 	ldr	x0, [sp, #56]
    47f0:	940004bb 	bl	5adc <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    47f4:	f9401fe2 	ldr	x2, [sp, #56]
    47f8:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    47fc:	912ea001 	add	x1, x0, #0xba8
    4800:	aa0203e0 	mov	x0, x2
    4804:	94000513 	bl	5c50 <sptPrintElapsedTime>
    4808:	fd005be0 	str	d0, [sp, #176]

	sptFreeTimer(timer);
    480c:	f9401fe0 	ldr	x0, [sp, #56]
    4810:	9400053b 	bl	5cfc <sptFreeTimer>
	sptFreeValueVector(&scratch);
    4814:	910103e0 	add	x0, sp, #0x40
    4818:	97fff799 	bl	267c <sptFreeValueVector>

	total_time = comp_time;
    481c:	fd405be0 	ldr	d0, [sp, #176]
    4820:	fd0057e0 	str	d0, [sp, #168]
	printf("[Total time]: %lf\n", total_time);
    4824:	fd4057e0 	ldr	d0, [sp, #168]
    4828:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    482c:	912f8000 	add	x0, x0, #0xbe0
    4830:	97fff2e0 	bl	13b0 <printf@plt>
	printf("\n");
    4834:	52800140 	mov	w0, #0xa                   	// #10
    4838:	97fff2ea 	bl	13e0 <putchar@plt>

	return 0;
    483c:	52800000 	mov	w0, #0x0                   	// #0
}
    4840:	a8d27bfd 	ldp	x29, x30, [sp], #288
    4844:	d65f03c0 	ret

0000000000004848 <sptMTTKRP_3D>:

int sptMTTKRP_3D(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode)
{
    4848:	a9b27bfd 	stp	x29, x30, [sp, #-224]!
    484c:	910003fd 	mov	x29, sp
    4850:	f90017e0 	str	x0, [sp, #40]
    4854:	f90013e1 	str	x1, [sp, #32]
    4858:	f9000fe2 	str	x2, [sp, #24]
    485c:	b90017e3 	str	w3, [sp, #20]
	sptIndex const nmodes = X->nmodes;
    4860:	f94017e0 	ldr	x0, [sp, #40]
    4864:	b9400000 	ldr	w0, [x0]
    4868:	b900cbe0 	str	w0, [sp, #200]
	sptNnzIndex const nnz = X->nnz;
    486c:	f94017e0 	ldr	x0, [sp, #40]
    4870:	f9400c00 	ldr	x0, [x0, #24]
    4874:	f90063e0 	str	x0, [sp, #192]
	sptIndex const * const ndims = X->ndims;
    4878:	f94017e0 	ldr	x0, [sp, #40]
    487c:	f9400800 	ldr	x0, [x0, #16]
    4880:	f9005fe0 	str	x0, [sp, #184]
	sptValue const * const restrict vals = X->values.data;
    4884:	f94017e0 	ldr	x0, [sp, #40]
    4888:	f9401c00 	ldr	x0, [x0, #56]
    488c:	f9005be0 	str	x0, [sp, #176]
	sptIndex const stride = mats[0]->stride;
    4890:	f94013e0 	ldr	x0, [sp, #32]
    4894:	f9400000 	ldr	x0, [x0]
    4898:	b9400c00 	ldr	w0, [x0, #12]
    489c:	b900afe0 	str	w0, [sp, #172]

	/* Check the mats. */
	sptAssert(nmodes ==3);
    48a0:	b940cbe0 	ldr	w0, [sp, #200]
    48a4:	71000c1f 	cmp	w0, #0x3
    48a8:	54000060 	b.eq	48b4 <sptMTTKRP_3D+0x6c>  // b.none
    48ac:	12800000 	mov	w0, #0xffffffff            	// #-1
    48b0:	97fff234 	bl	1180 <exit@plt>
	for(sptIndex i=0; i<nmodes; ++i) {
    48b4:	b900dfff 	str	wzr, [sp, #220]
    48b8:	14000035 	b	498c <sptMTTKRP_3D+0x144>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    48bc:	b940dfe0 	ldr	w0, [sp, #220]
    48c0:	d37df000 	lsl	x0, x0, #3
    48c4:	f94013e1 	ldr	x1, [sp, #32]
    48c8:	8b000020 	add	x0, x1, x0
    48cc:	f9400000 	ldr	x0, [x0]
    48d0:	b9400401 	ldr	w1, [x0, #4]
    48d4:	b940cbe0 	ldr	w0, [sp, #200]
    48d8:	d37df000 	lsl	x0, x0, #3
    48dc:	f94013e2 	ldr	x2, [sp, #32]
    48e0:	8b000040 	add	x0, x2, x0
    48e4:	f9400000 	ldr	x0, [x0]
    48e8:	b9400400 	ldr	w0, [x0, #4]
    48ec:	6b00003f 	cmp	w1, w0
    48f0:	54000180 	b.eq	4920 <sptMTTKRP_3D+0xd8>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    48f4:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    48f8:	912d6004 	add	x4, x0, #0xb58
    48fc:	52801263 	mov	w3, #0x93                  	// #147
    4900:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4904:	912e0002 	add	x2, x0, #0xb80
    4908:	52800041 	mov	w1, #0x2                   	// #2
    490c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4910:	912ea000 	add	x0, x0, #0xba8
    4914:	94000599 	bl	5f78 <spt_ComplainError>
    4918:	52800040 	mov	w0, #0x2                   	// #2
    491c:	140000e0 	b	4c9c <sptMTTKRP_3D+0x454>
		}
		if(mats[i]->nrows != ndims[i]) {
    4920:	b940dfe0 	ldr	w0, [sp, #220]
    4924:	d37df000 	lsl	x0, x0, #3
    4928:	f94013e1 	ldr	x1, [sp, #32]
    492c:	8b000020 	add	x0, x1, x0
    4930:	f9400000 	ldr	x0, [x0]
    4934:	b9400001 	ldr	w1, [x0]
    4938:	b940dfe0 	ldr	w0, [sp, #220]
    493c:	d37ef400 	lsl	x0, x0, #2
    4940:	f9405fe2 	ldr	x2, [sp, #184]
    4944:	8b000040 	add	x0, x2, x0
    4948:	b9400000 	ldr	w0, [x0]
    494c:	6b00003f 	cmp	w1, w0
    4950:	54000180 	b.eq	4980 <sptMTTKRP_3D+0x138>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Cpu SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4954:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4958:	912f0004 	add	x4, x0, #0xbc0
    495c:	528012c3 	mov	w3, #0x96                  	// #150
    4960:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4964:	912e0002 	add	x2, x0, #0xb80
    4968:	52800041 	mov	w1, #0x2                   	// #2
    496c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4970:	912ea000 	add	x0, x0, #0xba8
    4974:	94000581 	bl	5f78 <spt_ComplainError>
    4978:	52800040 	mov	w0, #0x2                   	// #2
    497c:	140000c8 	b	4c9c <sptMTTKRP_3D+0x454>
	for(sptIndex i=0; i<nmodes; ++i) {
    4980:	b940dfe0 	ldr	w0, [sp, #220]
    4984:	11000400 	add	w0, w0, #0x1
    4988:	b900dfe0 	str	w0, [sp, #220]
    498c:	b940dfe1 	ldr	w1, [sp, #220]
    4990:	b940cbe0 	ldr	w0, [sp, #200]
    4994:	6b00003f 	cmp	w1, w0
    4998:	54fff923 	b.cc	48bc <sptMTTKRP_3D+0x74>  // b.lo, b.ul, b.last
		}
	}


	sptIndex const tmpI = mats[mode]->nrows;
    499c:	b94017e0 	ldr	w0, [sp, #20]
    49a0:	d37df000 	lsl	x0, x0, #3
    49a4:	f94013e1 	ldr	x1, [sp, #32]
    49a8:	8b000020 	add	x0, x1, x0
    49ac:	f9400000 	ldr	x0, [x0]
    49b0:	b9400000 	ldr	w0, [x0]
    49b4:	b900abe0 	str	w0, [sp, #168]
	sptIndex const R = mats[mode]->ncols;
    49b8:	b94017e0 	ldr	w0, [sp, #20]
    49bc:	d37df000 	lsl	x0, x0, #3
    49c0:	f94013e1 	ldr	x1, [sp, #32]
    49c4:	8b000020 	add	x0, x1, x0
    49c8:	f9400000 	ldr	x0, [x0]
    49cc:	b9400400 	ldr	w0, [x0, #4]
    49d0:	b900a7e0 	str	w0, [sp, #164]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    49d4:	f94017e0 	ldr	x0, [sp, #40]
    49d8:	f9401002 	ldr	x2, [x0, #32]
    49dc:	b94017e1 	ldr	w1, [sp, #20]
    49e0:	aa0103e0 	mov	x0, x1
    49e4:	d37ff800 	lsl	x0, x0, #1
    49e8:	8b010000 	add	x0, x0, x1
    49ec:	d37df000 	lsl	x0, x0, #3
    49f0:	8b000040 	add	x0, x2, x0
    49f4:	f9400800 	ldr	x0, [x0, #16]
    49f8:	f9004fe0 	str	x0, [sp, #152]
	sptMatrix * const restrict M = mats[nmodes];
    49fc:	b940cbe0 	ldr	w0, [sp, #200]
    4a00:	d37df000 	lsl	x0, x0, #3
    4a04:	f94013e1 	ldr	x1, [sp, #32]
    4a08:	8b000020 	add	x0, x1, x0
    4a0c:	f9400000 	ldr	x0, [x0]
    4a10:	f9004be0 	str	x0, [sp, #144]
	sptValue * const restrict mvals = M->values;
    4a14:	f9404be0 	ldr	x0, [sp, #144]
    4a18:	f9400800 	ldr	x0, [x0, #16]
    4a1c:	f90047e0 	str	x0, [sp, #136]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4a20:	b940abe1 	ldr	w1, [sp, #168]
    4a24:	b940afe0 	ldr	w0, [sp, #172]
    4a28:	1b007c20 	mul	w0, w1, w0
    4a2c:	2a0003e0 	mov	w0, w0
    4a30:	d37ef400 	lsl	x0, x0, #2
    4a34:	aa0003e2 	mov	x2, x0
    4a38:	52800001 	mov	w1, #0x0                   	// #0
    4a3c:	f94047e0 	ldr	x0, [sp, #136]
    4a40:	97fff20c 	bl	1270 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
    4a44:	f9400fe0 	ldr	x0, [sp, #24]
    4a48:	b9400400 	ldr	w0, [x0, #4]
    4a4c:	b90087e0 	str	w0, [sp, #132]
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    4a50:	b94087e0 	ldr	w0, [sp, #132]
    4a54:	d37df000 	lsl	x0, x0, #3
    4a58:	f94013e1 	ldr	x1, [sp, #32]
    4a5c:	8b000020 	add	x0, x1, x0
    4a60:	f9400000 	ldr	x0, [x0]
    4a64:	f9003fe0 	str	x0, [sp, #120]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    4a68:	f94017e0 	ldr	x0, [sp, #40]
    4a6c:	f9401002 	ldr	x2, [x0, #32]
    4a70:	b94087e1 	ldr	w1, [sp, #132]
    4a74:	aa0103e0 	mov	x0, x1
    4a78:	d37ff800 	lsl	x0, x0, #1
    4a7c:	8b010000 	add	x0, x0, x1
    4a80:	d37df000 	lsl	x0, x0, #3
    4a84:	8b000040 	add	x0, x2, x0
    4a88:	f9400800 	ldr	x0, [x0, #16]
    4a8c:	f9003be0 	str	x0, [sp, #112]
	sptIndex times_mat_index_2 = mats_order[2];
    4a90:	f9400fe0 	ldr	x0, [sp, #24]
    4a94:	b9400800 	ldr	w0, [x0, #8]
    4a98:	b9006fe0 	str	w0, [sp, #108]
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    4a9c:	b9406fe0 	ldr	w0, [sp, #108]
    4aa0:	d37df000 	lsl	x0, x0, #3
    4aa4:	f94013e1 	ldr	x1, [sp, #32]
    4aa8:	8b000020 	add	x0, x1, x0
    4aac:	f9400000 	ldr	x0, [x0]
    4ab0:	f90033e0 	str	x0, [sp, #96]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    4ab4:	f94017e0 	ldr	x0, [sp, #40]
    4ab8:	f9401002 	ldr	x2, [x0, #32]
    4abc:	b9406fe1 	ldr	w1, [sp, #108]
    4ac0:	aa0103e0 	mov	x0, x1
    4ac4:	d37ff800 	lsl	x0, x0, #1
    4ac8:	8b010000 	add	x0, x0, x1
    4acc:	d37df000 	lsl	x0, x0, #3
    4ad0:	8b000040 	add	x0, x2, x0
    4ad4:	f9400800 	ldr	x0, [x0, #16]
    4ad8:	f9002fe0 	str	x0, [sp, #88]
	sptIndex mode_i;
	sptIndex tmp_i_1, tmp_i_2;
	sptValue entry;

	sptTimer timer;
	sptNewTimer(&timer, 0);
    4adc:	9100c3e0 	add	x0, sp, #0x30
    4ae0:	52800001 	mov	w1, #0x0                   	// #0
    4ae4:	9400039e 	bl	595c <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    4ae8:	f9401be0 	ldr	x0, [sp, #48]
    4aec:	940003bc 	bl	59dc <sptStartTimer>
	for(sptNnzIndex x=0; x<nnz; ++x) {
    4af0:	f9006bff 	str	xzr, [sp, #208]
    4af4:	14000053 	b	4c40 <sptMTTKRP_3D+0x3f8>
		mode_i = mode_ind[x];
    4af8:	f9406be0 	ldr	x0, [sp, #208]
    4afc:	d37ef400 	lsl	x0, x0, #2
    4b00:	f9404fe1 	ldr	x1, [sp, #152]
    4b04:	8b000020 	add	x0, x1, x0
    4b08:	b9400000 	ldr	w0, [x0]
    4b0c:	b90047e0 	str	w0, [sp, #68]
		tmp_i_1 = times_inds_1[x];
    4b10:	f9406be0 	ldr	x0, [sp, #208]
    4b14:	d37ef400 	lsl	x0, x0, #2
    4b18:	f9403be1 	ldr	x1, [sp, #112]
    4b1c:	8b000020 	add	x0, x1, x0
    4b20:	b9400000 	ldr	w0, [x0]
    4b24:	b90043e0 	str	w0, [sp, #64]
		tmp_i_2 = times_inds_2[x];
    4b28:	f9406be0 	ldr	x0, [sp, #208]
    4b2c:	d37ef400 	lsl	x0, x0, #2
    4b30:	f9402fe1 	ldr	x1, [sp, #88]
    4b34:	8b000020 	add	x0, x1, x0
    4b38:	b9400000 	ldr	w0, [x0]
    4b3c:	b9003fe0 	str	w0, [sp, #60]
		entry = vals[x];
    4b40:	f9406be0 	ldr	x0, [sp, #208]
    4b44:	d37ef400 	lsl	x0, x0, #2
    4b48:	f9405be1 	ldr	x1, [sp, #176]
    4b4c:	8b000020 	add	x0, x1, x0
    4b50:	bd400000 	ldr	s0, [x0]
    4b54:	bd003be0 	str	s0, [sp, #56]

		for(sptIndex r=0; r<R; ++r) {
    4b58:	b900cfff 	str	wzr, [sp, #204]
    4b5c:	14000032 	b	4c24 <sptMTTKRP_3D+0x3dc>
			mvals[mode_i * stride + r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    4b60:	b94047e1 	ldr	w1, [sp, #68]
    4b64:	b940afe0 	ldr	w0, [sp, #172]
    4b68:	1b007c21 	mul	w1, w1, w0
    4b6c:	b940cfe0 	ldr	w0, [sp, #204]
    4b70:	0b000020 	add	w0, w1, w0
    4b74:	2a0003e0 	mov	w0, w0
    4b78:	d37ef400 	lsl	x0, x0, #2
    4b7c:	f94047e1 	ldr	x1, [sp, #136]
    4b80:	8b000020 	add	x0, x1, x0
    4b84:	bd400001 	ldr	s1, [x0]
    4b88:	f9403fe0 	ldr	x0, [sp, #120]
    4b8c:	f9400801 	ldr	x1, [x0, #16]
    4b90:	b94043e2 	ldr	w2, [sp, #64]
    4b94:	b940afe0 	ldr	w0, [sp, #172]
    4b98:	1b007c42 	mul	w2, w2, w0
    4b9c:	b940cfe0 	ldr	w0, [sp, #204]
    4ba0:	0b000040 	add	w0, w2, w0
    4ba4:	2a0003e0 	mov	w0, w0
    4ba8:	d37ef400 	lsl	x0, x0, #2
    4bac:	8b000020 	add	x0, x1, x0
    4bb0:	bd400002 	ldr	s2, [x0]
    4bb4:	bd403be0 	ldr	s0, [sp, #56]
    4bb8:	1e200842 	fmul	s2, s2, s0
    4bbc:	f94033e0 	ldr	x0, [sp, #96]
    4bc0:	f9400801 	ldr	x1, [x0, #16]
    4bc4:	b9403fe2 	ldr	w2, [sp, #60]
    4bc8:	b940afe0 	ldr	w0, [sp, #172]
    4bcc:	1b007c42 	mul	w2, w2, w0
    4bd0:	b940cfe0 	ldr	w0, [sp, #204]
    4bd4:	0b000040 	add	w0, w2, w0
    4bd8:	2a0003e0 	mov	w0, w0
    4bdc:	d37ef400 	lsl	x0, x0, #2
    4be0:	8b000020 	add	x0, x1, x0
    4be4:	bd400000 	ldr	s0, [x0]
    4be8:	1e200840 	fmul	s0, s2, s0
    4bec:	b94047e1 	ldr	w1, [sp, #68]
    4bf0:	b940afe0 	ldr	w0, [sp, #172]
    4bf4:	1b007c21 	mul	w1, w1, w0
    4bf8:	b940cfe0 	ldr	w0, [sp, #204]
    4bfc:	0b000020 	add	w0, w1, w0
    4c00:	2a0003e0 	mov	w0, w0
    4c04:	d37ef400 	lsl	x0, x0, #2
    4c08:	f94047e1 	ldr	x1, [sp, #136]
    4c0c:	8b000020 	add	x0, x1, x0
    4c10:	1e202820 	fadd	s0, s1, s0
    4c14:	bd000000 	str	s0, [x0]
		for(sptIndex r=0; r<R; ++r) {
    4c18:	b940cfe0 	ldr	w0, [sp, #204]
    4c1c:	11000400 	add	w0, w0, #0x1
    4c20:	b900cfe0 	str	w0, [sp, #204]
    4c24:	b940cfe1 	ldr	w1, [sp, #204]
    4c28:	b940a7e0 	ldr	w0, [sp, #164]
    4c2c:	6b00003f 	cmp	w1, w0
    4c30:	54fff983 	b.cc	4b60 <sptMTTKRP_3D+0x318>  // b.lo, b.ul, b.last
	for(sptNnzIndex x=0; x<nnz; ++x) {
    4c34:	f9406be0 	ldr	x0, [sp, #208]
    4c38:	91000400 	add	x0, x0, #0x1
    4c3c:	f9006be0 	str	x0, [sp, #208]
    4c40:	f9406be1 	ldr	x1, [sp, #208]
    4c44:	f94063e0 	ldr	x0, [sp, #192]
    4c48:	eb00003f 	cmp	x1, x0
    4c4c:	54fff563 	b.cc	4af8 <sptMTTKRP_3D+0x2b0>  // b.lo, b.ul, b.last
		}
	}
	sptStopTimer(timer);
    4c50:	f9401be0 	ldr	x0, [sp, #48]
    4c54:	940003a2 	bl	5adc <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Cpu SpTns MTTKRP");
    4c58:	f9401be2 	ldr	x2, [sp, #48]
    4c5c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4c60:	912ea001 	add	x1, x0, #0xba8
    4c64:	aa0203e0 	mov	x0, x2
    4c68:	940003fa 	bl	5c50 <sptPrintElapsedTime>
    4c6c:	fd002be0 	str	d0, [sp, #80]
	sptFreeTimer(timer);
    4c70:	f9401be0 	ldr	x0, [sp, #48]
    4c74:	94000422 	bl	5cfc <sptFreeTimer>

	total_time = comp_time;
    4c78:	fd402be0 	ldr	d0, [sp, #80]
    4c7c:	fd0027e0 	str	d0, [sp, #72]
	printf("[Total time]: %lf\n", total_time);
    4c80:	fd4027e0 	ldr	d0, [sp, #72]
    4c84:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4c88:	912f8000 	add	x0, x0, #0xbe0
    4c8c:	97fff1c9 	bl	13b0 <printf@plt>
	printf("\n");
    4c90:	52800140 	mov	w0, #0xa                   	// #10
    4c94:	97fff1d3 	bl	13e0 <putchar@plt>

	return 0;
    4c98:	52800000 	mov	w0, #0x0                   	// #0
}
    4c9c:	a8ce7bfd 	ldp	x29, x30, [sp], #224
    4ca0:	d65f03c0 	ret

0000000000004ca4 <sptOmpMTTKRP>:
int sptOmpMTTKRP(sptSparseTensor const * const X,
								 sptMatrix * mats[],     // mats[nmodes] as temporary space.
								 sptIndex const mats_order[],    // Correspond to the mode order of X.
								 sptIndex const mode,
								 const int tk)
{
    4ca4:	a9b17bfd 	stp	x29, x30, [sp, #-240]!
    4ca8:	910003fd 	mov	x29, sp
    4cac:	f90017e0 	str	x0, [sp, #40]
    4cb0:	f90013e1 	str	x1, [sp, #32]
    4cb4:	f9000fe2 	str	x2, [sp, #24]
    4cb8:	b90017e3 	str	w3, [sp, #20]
    4cbc:	b90013e4 	str	w4, [sp, #16]
	sptIndex const nmodes = X->nmodes;
    4cc0:	f94017e0 	ldr	x0, [sp, #40]
    4cc4:	b9400000 	ldr	w0, [x0]
    4cc8:	b900ebe0 	str	w0, [sp, #232]

	if(nmodes == 3) {
    4ccc:	b940ebe0 	ldr	w0, [sp, #232]
    4cd0:	71000c1f 	cmp	w0, #0x3
    4cd4:	540001a1 	b.ne	4d08 <sptOmpMTTKRP+0x64>  // b.any
		sptAssert(sptOmpMTTKRP_3D(X, mats, mats_order, mode, tk) == 0);
    4cd8:	b94013e4 	ldr	w4, [sp, #16]
    4cdc:	b94017e3 	ldr	w3, [sp, #20]
    4ce0:	f9400fe2 	ldr	x2, [sp, #24]
    4ce4:	f94013e1 	ldr	x1, [sp, #32]
    4ce8:	f94017e0 	ldr	x0, [sp, #40]
    4cec:	940000ac 	bl	4f9c <sptOmpMTTKRP_3D>
    4cf0:	7100001f 	cmp	w0, #0x0
    4cf4:	54000060 	b.eq	4d00 <sptOmpMTTKRP+0x5c>  // b.none
    4cf8:	12800000 	mov	w0, #0xffffffff            	// #-1
    4cfc:	97fff121 	bl	1180 <exit@plt>
		return 0;
    4d00:	52800000 	mov	w0, #0x0                   	// #0
    4d04:	140000a4 	b	4f94 <sptOmpMTTKRP+0x2f0>
	}

	sptNnzIndex const nnz = X->nnz;
    4d08:	f94017e0 	ldr	x0, [sp, #40]
    4d0c:	f9400c00 	ldr	x0, [x0, #24]
    4d10:	f90073e0 	str	x0, [sp, #224]
	sptIndex const * const ndims = X->ndims;
    4d14:	f94017e0 	ldr	x0, [sp, #40]
    4d18:	f9400800 	ldr	x0, [x0, #16]
    4d1c:	f9006fe0 	str	x0, [sp, #216]
	sptValue const * const vals = X->values.data;
    4d20:	f94017e0 	ldr	x0, [sp, #40]
    4d24:	f9401c00 	ldr	x0, [x0, #56]
    4d28:	f9006be0 	str	x0, [sp, #208]
	sptIndex const stride = mats[0]->stride;
    4d2c:	f94013e0 	ldr	x0, [sp, #32]
    4d30:	f9400000 	ldr	x0, [x0]
    4d34:	b9400c00 	ldr	w0, [x0, #12]
    4d38:	b900cfe0 	str	w0, [sp, #204]

	/* Check the mats. */
	for(sptIndex i=0; i<nmodes; ++i) {
    4d3c:	b900efff 	str	wzr, [sp, #236]
    4d40:	14000035 	b	4e14 <sptOmpMTTKRP+0x170>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    4d44:	b940efe0 	ldr	w0, [sp, #236]
    4d48:	d37df000 	lsl	x0, x0, #3
    4d4c:	f94013e1 	ldr	x1, [sp, #32]
    4d50:	8b000020 	add	x0, x1, x0
    4d54:	f9400000 	ldr	x0, [x0]
    4d58:	b9400401 	ldr	w1, [x0, #4]
    4d5c:	b940ebe0 	ldr	w0, [sp, #232]
    4d60:	d37df000 	lsl	x0, x0, #3
    4d64:	f94013e2 	ldr	x2, [sp, #32]
    4d68:	8b000040 	add	x0, x2, x0
    4d6c:	f9400000 	ldr	x0, [x0]
    4d70:	b9400400 	ldr	w0, [x0, #4]
    4d74:	6b00003f 	cmp	w1, w0
    4d78:	54000180 	b.eq	4da8 <sptOmpMTTKRP+0x104>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    4d7c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4d80:	912fe004 	add	x4, x0, #0xbf8
    4d84:	52800823 	mov	w3, #0x41                  	// #65
    4d88:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4d8c:	91308002 	add	x2, x0, #0xc20
    4d90:	52800041 	mov	w1, #0x2                   	// #2
    4d94:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4d98:	91312000 	add	x0, x0, #0xc48
    4d9c:	94000477 	bl	5f78 <spt_ComplainError>
    4da0:	52800040 	mov	w0, #0x2                   	// #2
    4da4:	1400007c 	b	4f94 <sptOmpMTTKRP+0x2f0>
		}
		if(mats[i]->nrows != ndims[i]) {
    4da8:	b940efe0 	ldr	w0, [sp, #236]
    4dac:	d37df000 	lsl	x0, x0, #3
    4db0:	f94013e1 	ldr	x1, [sp, #32]
    4db4:	8b000020 	add	x0, x1, x0
    4db8:	f9400000 	ldr	x0, [x0]
    4dbc:	b9400001 	ldr	w1, [x0]
    4dc0:	b940efe0 	ldr	w0, [sp, #236]
    4dc4:	d37ef400 	lsl	x0, x0, #2
    4dc8:	f9406fe2 	ldr	x2, [sp, #216]
    4dcc:	8b000040 	add	x0, x2, x0
    4dd0:	b9400000 	ldr	w0, [x0]
    4dd4:	6b00003f 	cmp	w1, w0
    4dd8:	54000180 	b.eq	4e08 <sptOmpMTTKRP+0x164>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    4ddc:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4de0:	91318004 	add	x4, x0, #0xc60
    4de4:	52800883 	mov	w3, #0x44                  	// #68
    4de8:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4dec:	91308002 	add	x2, x0, #0xc20
    4df0:	52800041 	mov	w1, #0x2                   	// #2
    4df4:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4df8:	91312000 	add	x0, x0, #0xc48
    4dfc:	9400045f 	bl	5f78 <spt_ComplainError>
    4e00:	52800040 	mov	w0, #0x2                   	// #2
    4e04:	14000064 	b	4f94 <sptOmpMTTKRP+0x2f0>
	for(sptIndex i=0; i<nmodes; ++i) {
    4e08:	b940efe0 	ldr	w0, [sp, #236]
    4e0c:	11000400 	add	w0, w0, #0x1
    4e10:	b900efe0 	str	w0, [sp, #236]
    4e14:	b940efe1 	ldr	w1, [sp, #236]
    4e18:	b940ebe0 	ldr	w0, [sp, #232]
    4e1c:	6b00003f 	cmp	w1, w0
    4e20:	54fff923 	b.cc	4d44 <sptOmpMTTKRP+0xa0>  // b.lo, b.ul, b.last
		}
	}

	sptIndex const tmpI = mats[mode]->nrows;
    4e24:	b94017e0 	ldr	w0, [sp, #20]
    4e28:	d37df000 	lsl	x0, x0, #3
    4e2c:	f94013e1 	ldr	x1, [sp, #32]
    4e30:	8b000020 	add	x0, x1, x0
    4e34:	f9400000 	ldr	x0, [x0]
    4e38:	b9400000 	ldr	w0, [x0]
    4e3c:	b900cbe0 	str	w0, [sp, #200]
	sptIndex const R = mats[mode]->ncols;
    4e40:	b94017e0 	ldr	w0, [sp, #20]
    4e44:	d37df000 	lsl	x0, x0, #3
    4e48:	f94013e1 	ldr	x1, [sp, #32]
    4e4c:	8b000020 	add	x0, x1, x0
    4e50:	f9400000 	ldr	x0, [x0]
    4e54:	b9400400 	ldr	w0, [x0, #4]
    4e58:	b900c7e0 	str	w0, [sp, #196]
	sptIndex const * const mode_ind = X->inds[mode].data;
    4e5c:	f94017e0 	ldr	x0, [sp, #40]
    4e60:	f9401002 	ldr	x2, [x0, #32]
    4e64:	b94017e1 	ldr	w1, [sp, #20]
    4e68:	aa0103e0 	mov	x0, x1
    4e6c:	d37ff800 	lsl	x0, x0, #1
    4e70:	8b010000 	add	x0, x0, x1
    4e74:	d37df000 	lsl	x0, x0, #3
    4e78:	8b000040 	add	x0, x2, x0
    4e7c:	f9400800 	ldr	x0, [x0, #16]
    4e80:	f9005fe0 	str	x0, [sp, #184]
	sptValue * const restrict mvals = mats[nmodes]->values;
    4e84:	b940ebe0 	ldr	w0, [sp, #232]
    4e88:	d37df000 	lsl	x0, x0, #3
    4e8c:	f94013e1 	ldr	x1, [sp, #32]
    4e90:	8b000020 	add	x0, x1, x0
    4e94:	f9400000 	ldr	x0, [x0]
    4e98:	f9400800 	ldr	x0, [x0, #16]
    4e9c:	f9005be0 	str	x0, [sp, #176]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    4ea0:	b940cbe1 	ldr	w1, [sp, #200]
    4ea4:	b940cfe0 	ldr	w0, [sp, #204]
    4ea8:	1b007c20 	mul	w0, w1, w0
    4eac:	2a0003e0 	mov	w0, w0
    4eb0:	d37ef400 	lsl	x0, x0, #2
    4eb4:	aa0003e2 	mov	x2, x0
    4eb8:	52800001 	mov	w1, #0x0                   	// #0
    4ebc:	f9405be0 	ldr	x0, [sp, #176]
    4ec0:	97fff0ec 	bl	1270 <memset@plt>

	sptTimer timer;
	sptNewTimer(&timer, 0);
    4ec4:	9100e3e0 	add	x0, sp, #0x38
    4ec8:	52800001 	mov	w1, #0x0                   	// #0
    4ecc:	940002a4 	bl	595c <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    4ed0:	f9401fe0 	ldr	x0, [sp, #56]
    4ed4:	940002c2 	bl	59dc <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    4ed8:	f9405be0 	ldr	x0, [sp, #176]
    4edc:	f9003be0 	str	x0, [sp, #112]
    4ee0:	f9405fe0 	ldr	x0, [sp, #184]
    4ee4:	f90037e0 	str	x0, [sp, #104]
    4ee8:	b940c7e0 	ldr	w0, [sp, #196]
    4eec:	b90083e0 	str	w0, [sp, #128]
    4ef0:	b940cfe0 	ldr	w0, [sp, #204]
    4ef4:	b9007fe0 	str	w0, [sp, #124]
    4ef8:	f9406be0 	ldr	x0, [sp, #208]
    4efc:	f90033e0 	str	x0, [sp, #96]
    4f00:	f94073e0 	ldr	x0, [sp, #224]
    4f04:	f9002fe0 	str	x0, [sp, #88]
    4f08:	b940ebe0 	ldr	w0, [sp, #232]
    4f0c:	b9007be0 	str	w0, [sp, #120]
    4f10:	f9400fe0 	ldr	x0, [sp, #24]
    4f14:	f9002be0 	str	x0, [sp, #80]
    4f18:	f94013e0 	ldr	x0, [sp, #32]
    4f1c:	f90027e0 	str	x0, [sp, #72]
    4f20:	f94017e0 	ldr	x0, [sp, #40]
    4f24:	f90023e0 	str	x0, [sp, #64]
    4f28:	b94013e1 	ldr	w1, [sp, #16]
    4f2c:	910103e0 	add	x0, sp, #0x40
    4f30:	52800003 	mov	w3, #0x0                   	// #0
    4f34:	2a0103e2 	mov	w2, w1
    4f38:	aa0003e1 	mov	x1, x0
    4f3c:	b0000000 	adrp	x0, 5000 <sptOmpMTTKRP_3D+0x64>
    4f40:	910c1000 	add	x0, x0, #0x304
    4f44:	97fff123 	bl	13d0 <GOMP_parallel@plt>
		}

		sptFreeValueVector(&scratch);
	}   // End loop nnzs

	sptStopTimer(timer);
    4f48:	f9401fe0 	ldr	x0, [sp, #56]
    4f4c:	940002e4 	bl	5adc <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    4f50:	f9401fe2 	ldr	x2, [sp, #56]
    4f54:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4f58:	91312001 	add	x1, x0, #0xc48
    4f5c:	aa0203e0 	mov	x0, x2
    4f60:	9400033c 	bl	5c50 <sptPrintElapsedTime>
    4f64:	fd0057e0 	str	d0, [sp, #168]

	sptFreeTimer(timer);
    4f68:	f9401fe0 	ldr	x0, [sp, #56]
    4f6c:	94000364 	bl	5cfc <sptFreeTimer>

	total_time = comp_time;
    4f70:	fd4057e0 	ldr	d0, [sp, #168]
    4f74:	fd0053e0 	str	d0, [sp, #160]
	printf("[Total time]: %lf\n", total_time);
    4f78:	fd4053e0 	ldr	d0, [sp, #160]
    4f7c:	d0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    4f80:	91320000 	add	x0, x0, #0xc80
    4f84:	97fff10b 	bl	13b0 <printf@plt>
	printf("\n");
    4f88:	52800140 	mov	w0, #0xa                   	// #10
    4f8c:	97fff115 	bl	13e0 <putchar@plt>

	return 0;
    4f90:	52800000 	mov	w0, #0x0                   	// #0
}
    4f94:	a8cf7bfd 	ldp	x29, x30, [sp], #240
    4f98:	d65f03c0 	ret

0000000000004f9c <sptOmpMTTKRP_3D>:
int sptOmpMTTKRP_3D(sptSparseTensor const * const X,
										sptMatrix * mats[],     // mats[nmodes] as temporary space.
										sptIndex const mats_order[],    // Correspond to the mode order of X.
										sptIndex const mode,
										const int tk)
{
    4f9c:	a9b07bfd 	stp	x29, x30, [sp, #-256]!
    4fa0:	910003fd 	mov	x29, sp
    4fa4:	f90017e0 	str	x0, [sp, #40]
    4fa8:	f90013e1 	str	x1, [sp, #32]
    4fac:	f9000fe2 	str	x2, [sp, #24]
    4fb0:	b90017e3 	str	w3, [sp, #20]
    4fb4:	b90013e4 	str	w4, [sp, #16]
	sptIndex const nmodes = X->nmodes;
    4fb8:	f94017e0 	ldr	x0, [sp, #40]
    4fbc:	b9400000 	ldr	w0, [x0]
    4fc0:	b900fbe0 	str	w0, [sp, #248]
	sptNnzIndex const nnz = X->nnz;
    4fc4:	f94017e0 	ldr	x0, [sp, #40]
    4fc8:	f9400c00 	ldr	x0, [x0, #24]
    4fcc:	f9007be0 	str	x0, [sp, #240]
	sptIndex const * const ndims = X->ndims;
    4fd0:	f94017e0 	ldr	x0, [sp, #40]
    4fd4:	f9400800 	ldr	x0, [x0, #16]
    4fd8:	f90077e0 	str	x0, [sp, #232]
	sptValue const * const restrict vals = X->values.data;
    4fdc:	f94017e0 	ldr	x0, [sp, #40]
    4fe0:	f9401c00 	ldr	x0, [x0, #56]
    4fe4:	f90073e0 	str	x0, [sp, #224]
	sptIndex const stride = mats[0]->stride;
    4fe8:	f94013e0 	ldr	x0, [sp, #32]
    4fec:	f9400000 	ldr	x0, [x0]
    4ff0:	b9400c00 	ldr	w0, [x0, #12]
    4ff4:	b900dfe0 	str	w0, [sp, #220]

	/* Check the mats. */
	sptAssert(nmodes ==3);
    4ff8:	b940fbe0 	ldr	w0, [sp, #248]
    4ffc:	71000c1f 	cmp	w0, #0x3
    5000:	54000060 	b.eq	500c <sptOmpMTTKRP_3D+0x70>  // b.none
    5004:	12800000 	mov	w0, #0xffffffff            	// #-1
    5008:	97fff05e 	bl	1180 <exit@plt>
	for(sptIndex i=0; i<nmodes; ++i) {
    500c:	b900ffff 	str	wzr, [sp, #252]
    5010:	14000035 	b	50e4 <sptOmpMTTKRP_3D+0x148>
		if(mats[i]->ncols != mats[nmodes]->ncols) {
    5014:	b940ffe0 	ldr	w0, [sp, #252]
    5018:	d37df000 	lsl	x0, x0, #3
    501c:	f94013e1 	ldr	x1, [sp, #32]
    5020:	8b000020 	add	x0, x1, x0
    5024:	f9400000 	ldr	x0, [x0]
    5028:	b9400401 	ldr	w1, [x0, #4]
    502c:	b940fbe0 	ldr	w0, [sp, #248]
    5030:	d37df000 	lsl	x0, x0, #3
    5034:	f94013e2 	ldr	x2, [sp, #32]
    5038:	8b000040 	add	x0, x2, x0
    503c:	f9400000 	ldr	x0, [x0]
    5040:	b9400400 	ldr	w0, [x0, #4]
    5044:	6b00003f 	cmp	w1, w0
    5048:	54000180 	b.eq	5078 <sptOmpMTTKRP_3D+0xdc>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->cols != mats[nmodes]->ncols");
    504c:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5050:	912fe004 	add	x4, x0, #0xbf8
    5054:	528012c3 	mov	w3, #0x96                  	// #150
    5058:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    505c:	91308002 	add	x2, x0, #0xc20
    5060:	52800041 	mov	w1, #0x2                   	// #2
    5064:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5068:	91312000 	add	x0, x0, #0xc48
    506c:	940003c3 	bl	5f78 <spt_ComplainError>
    5070:	52800040 	mov	w0, #0x2                   	// #2
    5074:	140000a2 	b	52fc <sptOmpMTTKRP_3D+0x360>
		}
		if(mats[i]->nrows != ndims[i]) {
    5078:	b940ffe0 	ldr	w0, [sp, #252]
    507c:	d37df000 	lsl	x0, x0, #3
    5080:	f94013e1 	ldr	x1, [sp, #32]
    5084:	8b000020 	add	x0, x1, x0
    5088:	f9400000 	ldr	x0, [x0]
    508c:	b9400001 	ldr	w1, [x0]
    5090:	b940ffe0 	ldr	w0, [sp, #252]
    5094:	d37ef400 	lsl	x0, x0, #2
    5098:	f94077e2 	ldr	x2, [sp, #232]
    509c:	8b000040 	add	x0, x2, x0
    50a0:	b9400000 	ldr	w0, [x0]
    50a4:	6b00003f 	cmp	w1, w0
    50a8:	54000180 	b.eq	50d8 <sptOmpMTTKRP_3D+0x13c>  // b.none
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "Omp SpTns MTTKRP", "mats[i]->nrows != ndims[i]");
    50ac:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    50b0:	91318004 	add	x4, x0, #0xc60
    50b4:	52801323 	mov	w3, #0x99                  	// #153
    50b8:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    50bc:	91308002 	add	x2, x0, #0xc20
    50c0:	52800041 	mov	w1, #0x2                   	// #2
    50c4:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    50c8:	91312000 	add	x0, x0, #0xc48
    50cc:	940003ab 	bl	5f78 <spt_ComplainError>
    50d0:	52800040 	mov	w0, #0x2                   	// #2
    50d4:	1400008a 	b	52fc <sptOmpMTTKRP_3D+0x360>
	for(sptIndex i=0; i<nmodes; ++i) {
    50d8:	b940ffe0 	ldr	w0, [sp, #252]
    50dc:	11000400 	add	w0, w0, #0x1
    50e0:	b900ffe0 	str	w0, [sp, #252]
    50e4:	b940ffe1 	ldr	w1, [sp, #252]
    50e8:	b940fbe0 	ldr	w0, [sp, #248]
    50ec:	6b00003f 	cmp	w1, w0
    50f0:	54fff923 	b.cc	5014 <sptOmpMTTKRP_3D+0x78>  // b.lo, b.ul, b.last
		}
	}

	sptIndex const tmpI = mats[mode]->nrows;
    50f4:	b94017e0 	ldr	w0, [sp, #20]
    50f8:	d37df000 	lsl	x0, x0, #3
    50fc:	f94013e1 	ldr	x1, [sp, #32]
    5100:	8b000020 	add	x0, x1, x0
    5104:	f9400000 	ldr	x0, [x0]
    5108:	b9400000 	ldr	w0, [x0]
    510c:	b900dbe0 	str	w0, [sp, #216]
	sptIndex const R = mats[mode]->ncols;
    5110:	b94017e0 	ldr	w0, [sp, #20]
    5114:	d37df000 	lsl	x0, x0, #3
    5118:	f94013e1 	ldr	x1, [sp, #32]
    511c:	8b000020 	add	x0, x1, x0
    5120:	f9400000 	ldr	x0, [x0]
    5124:	b9400400 	ldr	w0, [x0, #4]
    5128:	b900d7e0 	str	w0, [sp, #212]
	sptIndex const * const restrict mode_ind = X->inds[mode].data;
    512c:	f94017e0 	ldr	x0, [sp, #40]
    5130:	f9401002 	ldr	x2, [x0, #32]
    5134:	b94017e1 	ldr	w1, [sp, #20]
    5138:	aa0103e0 	mov	x0, x1
    513c:	d37ff800 	lsl	x0, x0, #1
    5140:	8b010000 	add	x0, x0, x1
    5144:	d37df000 	lsl	x0, x0, #3
    5148:	8b000040 	add	x0, x2, x0
    514c:	f9400800 	ldr	x0, [x0, #16]
    5150:	f90067e0 	str	x0, [sp, #200]
	sptValue * const restrict mvals = mats[nmodes]->values;
    5154:	b940fbe0 	ldr	w0, [sp, #248]
    5158:	d37df000 	lsl	x0, x0, #3
    515c:	f94013e1 	ldr	x1, [sp, #32]
    5160:	8b000020 	add	x0, x1, x0
    5164:	f9400000 	ldr	x0, [x0]
    5168:	f9400800 	ldr	x0, [x0, #16]
    516c:	f90063e0 	str	x0, [sp, #192]
	memset(mvals, 0, tmpI*stride*sizeof(sptValue));
    5170:	b940dbe1 	ldr	w1, [sp, #216]
    5174:	b940dfe0 	ldr	w0, [sp, #220]
    5178:	1b007c20 	mul	w0, w1, w0
    517c:	2a0003e0 	mov	w0, w0
    5180:	d37ef400 	lsl	x0, x0, #2
    5184:	aa0003e2 	mov	x2, x0
    5188:	52800001 	mov	w1, #0x0                   	// #0
    518c:	f94063e0 	ldr	x0, [sp, #192]
    5190:	97fff038 	bl	1270 <memset@plt>

	sptIndex times_mat_index_1 = mats_order[1];
    5194:	f9400fe0 	ldr	x0, [sp, #24]
    5198:	b9400400 	ldr	w0, [x0, #4]
    519c:	b900bfe0 	str	w0, [sp, #188]
	sptMatrix * restrict times_mat_1 = mats[times_mat_index_1];
    51a0:	b940bfe0 	ldr	w0, [sp, #188]
    51a4:	d37df000 	lsl	x0, x0, #3
    51a8:	f94013e1 	ldr	x1, [sp, #32]
    51ac:	8b000020 	add	x0, x1, x0
    51b0:	f9400000 	ldr	x0, [x0]
    51b4:	f9005be0 	str	x0, [sp, #176]
	sptIndex * restrict times_inds_1 = X->inds[times_mat_index_1].data;
    51b8:	f94017e0 	ldr	x0, [sp, #40]
    51bc:	f9401002 	ldr	x2, [x0, #32]
    51c0:	b940bfe1 	ldr	w1, [sp, #188]
    51c4:	aa0103e0 	mov	x0, x1
    51c8:	d37ff800 	lsl	x0, x0, #1
    51cc:	8b010000 	add	x0, x0, x1
    51d0:	d37df000 	lsl	x0, x0, #3
    51d4:	8b000040 	add	x0, x2, x0
    51d8:	f9400800 	ldr	x0, [x0, #16]
    51dc:	f90057e0 	str	x0, [sp, #168]
	sptIndex times_mat_index_2 = mats_order[2];
    51e0:	f9400fe0 	ldr	x0, [sp, #24]
    51e4:	b9400800 	ldr	w0, [x0, #8]
    51e8:	b900a7e0 	str	w0, [sp, #164]
	sptMatrix * restrict times_mat_2 = mats[times_mat_index_2];
    51ec:	b940a7e0 	ldr	w0, [sp, #164]
    51f0:	d37df000 	lsl	x0, x0, #3
    51f4:	f94013e1 	ldr	x1, [sp, #32]
    51f8:	8b000020 	add	x0, x1, x0
    51fc:	f9400000 	ldr	x0, [x0]
    5200:	f9004fe0 	str	x0, [sp, #152]
	sptIndex * restrict times_inds_2 = X->inds[times_mat_index_2].data;
    5204:	f94017e0 	ldr	x0, [sp, #40]
    5208:	f9401002 	ldr	x2, [x0, #32]
    520c:	b940a7e1 	ldr	w1, [sp, #164]
    5210:	aa0103e0 	mov	x0, x1
    5214:	d37ff800 	lsl	x0, x0, #1
    5218:	8b010000 	add	x0, x0, x1
    521c:	d37df000 	lsl	x0, x0, #3
    5220:	8b000040 	add	x0, x2, x0
    5224:	f9400800 	ldr	x0, [x0, #16]
    5228:	f9004be0 	str	x0, [sp, #144]

	sptTimer timer;
	sptNewTimer(&timer, 0);
    522c:	9100c3e0 	add	x0, sp, #0x30
    5230:	52800001 	mov	w1, #0x0                   	// #0
    5234:	940001ca 	bl	595c <sptNewTimer>
	double comp_time, total_time;

	sptStartTimer(timer);
    5238:	f9401be0 	ldr	x0, [sp, #48]
    523c:	940001e8 	bl	59dc <sptStartTimer>
#pragma omp parallel for schedule(static) num_threads(tk)
    5240:	f9404be0 	ldr	x0, [sp, #144]
    5244:	f9003be0 	str	x0, [sp, #112]
    5248:	f9404fe0 	ldr	x0, [sp, #152]
    524c:	f90037e0 	str	x0, [sp, #104]
    5250:	f94057e0 	ldr	x0, [sp, #168]
    5254:	f90033e0 	str	x0, [sp, #96]
    5258:	f9405be0 	ldr	x0, [sp, #176]
    525c:	f9002fe0 	str	x0, [sp, #88]
    5260:	f94063e0 	ldr	x0, [sp, #192]
    5264:	f9002be0 	str	x0, [sp, #80]
    5268:	f94067e0 	ldr	x0, [sp, #200]
    526c:	f90027e0 	str	x0, [sp, #72]
    5270:	b940d7e0 	ldr	w0, [sp, #212]
    5274:	b9007fe0 	str	w0, [sp, #124]
    5278:	b940dfe0 	ldr	w0, [sp, #220]
    527c:	b9007be0 	str	w0, [sp, #120]
    5280:	f94073e0 	ldr	x0, [sp, #224]
    5284:	f90023e0 	str	x0, [sp, #64]
    5288:	f9407be0 	ldr	x0, [sp, #240]
    528c:	f9001fe0 	str	x0, [sp, #56]
    5290:	b94013e1 	ldr	w1, [sp, #16]
    5294:	9100e3e0 	add	x0, sp, #0x38
    5298:	52800003 	mov	w3, #0x0                   	// #0
    529c:	2a0103e2 	mov	w2, w1
    52a0:	aa0003e1 	mov	x1, x0
    52a4:	90000000 	adrp	x0, 5000 <sptOmpMTTKRP_3D+0x64>
    52a8:	911bd000 	add	x0, x0, #0x6f4
    52ac:	97fff049 	bl	13d0 <GOMP_parallel@plt>
		for(sptIndex r=0; r<R; ++r) {
#pragma omp atomic update
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
		}
	}
	sptStopTimer(timer);
    52b0:	f9401be0 	ldr	x0, [sp, #48]
    52b4:	9400020a 	bl	5adc <sptStopTimer>
	comp_time = sptPrintElapsedTime(timer, "Omp SpTns MTTKRP");
    52b8:	f9401be2 	ldr	x2, [sp, #48]
    52bc:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    52c0:	91312001 	add	x1, x0, #0xc48
    52c4:	aa0203e0 	mov	x0, x2
    52c8:	94000262 	bl	5c50 <sptPrintElapsedTime>
    52cc:	fd0047e0 	str	d0, [sp, #136]

	sptFreeTimer(timer);
    52d0:	f9401be0 	ldr	x0, [sp, #48]
    52d4:	9400028a 	bl	5cfc <sptFreeTimer>

	total_time = comp_time;
    52d8:	fd4047e0 	ldr	d0, [sp, #136]
    52dc:	fd0043e0 	str	d0, [sp, #128]
	printf("[Total time]: %lf\n", total_time);
    52e0:	fd4043e0 	ldr	d0, [sp, #128]
    52e4:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    52e8:	91320000 	add	x0, x0, #0xc80
    52ec:	97fff031 	bl	13b0 <printf@plt>
	printf("\n");
    52f0:	52800140 	mov	w0, #0xa                   	// #10
    52f4:	97fff03b 	bl	13e0 <putchar@plt>

	return 0;
    52f8:	52800000 	mov	w0, #0x0                   	// #0
}
    52fc:	a8d07bfd 	ldp	x29, x30, [sp], #256
    5300:	d65f03c0 	ret

0000000000005304 <sptOmpMTTKRP._omp_fn.0>:
#pragma omp parallel for schedule(static) num_threads(tk)
    5304:	a9b17bfd 	stp	x29, x30, [sp, #-240]!
    5308:	910003fd 	mov	x29, sp
    530c:	a90153f3 	stp	x19, x20, [sp, #16]
    5310:	f90013f5 	str	x21, [sp, #32]
    5314:	fd0017e8 	str	d8, [sp, #40]
    5318:	f9001fe0 	str	x0, [sp, #56]
    531c:	f9401fe0 	ldr	x0, [sp, #56]
    5320:	f9401800 	ldr	x0, [x0, #48]
    5324:	f9006be0 	str	x0, [sp, #208]
    5328:	f9401fe0 	ldr	x0, [sp, #56]
    532c:	f9401400 	ldr	x0, [x0, #40]
    5330:	f90067e0 	str	x0, [sp, #200]
    5334:	f9401fe0 	ldr	x0, [sp, #56]
    5338:	b9404000 	ldr	w0, [x0, #64]
    533c:	b900c7e0 	str	w0, [sp, #196]
    5340:	f9401fe0 	ldr	x0, [sp, #56]
    5344:	b9403c00 	ldr	w0, [x0, #60]
    5348:	b900c3e0 	str	w0, [sp, #192]
    534c:	f9401fe0 	ldr	x0, [sp, #56]
    5350:	f9401000 	ldr	x0, [x0, #32]
    5354:	f9005fe0 	str	x0, [sp, #184]
    5358:	f9401fe0 	ldr	x0, [sp, #56]
    535c:	f9400c00 	ldr	x0, [x0, #24]
    5360:	f9005be0 	str	x0, [sp, #176]
    5364:	f9401fe0 	ldr	x0, [sp, #56]
    5368:	b9403800 	ldr	w0, [x0, #56]
    536c:	b900afe0 	str	w0, [sp, #172]
    5370:	f9401fe0 	ldr	x0, [sp, #56]
    5374:	f9400800 	ldr	x0, [x0, #16]
    5378:	f90053e0 	str	x0, [sp, #160]
    537c:	f9401fe0 	ldr	x0, [sp, #56]
    5380:	f9400400 	ldr	x0, [x0, #8]
    5384:	f9004fe0 	str	x0, [sp, #152]
    5388:	f9401fe0 	ldr	x0, [sp, #56]
    538c:	f9400000 	ldr	x0, [x0]
    5390:	f9004be0 	str	x0, [sp, #144]
    5394:	f9405bf3 	ldr	x19, [sp, #176]
    5398:	f100027f 	cmp	x19, #0x0
    539c:	54001a00 	b.eq	56dc <sptOmpMTTKRP._omp_fn.0+0x3d8>  // b.none
    53a0:	97ffefc4 	bl	12b0 <omp_get_num_threads@plt>
    53a4:	93407c14 	sxtw	x20, w0
    53a8:	97ffef8e 	bl	11e0 <omp_get_thread_num@plt>
    53ac:	93407c02 	sxtw	x2, w0
    53b0:	9ad40a60 	udiv	x0, x19, x20
    53b4:	9ad40a61 	udiv	x1, x19, x20
    53b8:	9b147c21 	mul	x1, x1, x20
    53bc:	cb010261 	sub	x1, x19, x1
    53c0:	eb01005f 	cmp	x2, x1
    53c4:	54001863 	b.cc	56d0 <sptOmpMTTKRP._omp_fn.0+0x3cc>  // b.lo, b.ul, b.last
    53c8:	9b027c02 	mul	x2, x0, x2
    53cc:	8b010041 	add	x1, x2, x1
    53d0:	8b000034 	add	x20, x1, x0
    53d4:	eb14003f 	cmp	x1, x20
    53d8:	54001822 	b.cs	56dc <sptOmpMTTKRP._omp_fn.0+0x3d8>  // b.hs, b.nlast
    53dc:	f90077e1 	str	x1, [sp, #232]
		sptNewValueVector(&scratch, R, R);
    53e0:	b940c7e1 	ldr	w1, [sp, #196]
    53e4:	b940c7e2 	ldr	w2, [sp, #196]
    53e8:	910123e0 	add	x0, sp, #0x48
    53ec:	97fff3a6 	bl	2284 <sptNewValueVector>
		sptConstantValueVector(&scratch, 0);
    53f0:	910123e0 	add	x0, sp, #0x48
    53f4:	0f000400 	movi	v0.2s, #0x0
    53f8:	97fff3ee 	bl	23b0 <sptConstantValueVector>
		sptIndex times_mat_index = mats_order[1];
    53fc:	f94053e0 	ldr	x0, [sp, #160]
    5400:	b9400400 	ldr	w0, [x0, #4]
    5404:	b9008fe0 	str	w0, [sp, #140]
		sptMatrix * times_mat = mats[times_mat_index];
    5408:	b9408fe0 	ldr	w0, [sp, #140]
    540c:	d37df000 	lsl	x0, x0, #3
    5410:	f9404fe1 	ldr	x1, [sp, #152]
    5414:	8b000020 	add	x0, x1, x0
    5418:	f9400000 	ldr	x0, [x0]
    541c:	f90043e0 	str	x0, [sp, #128]
		sptIndex * times_inds = X->inds[times_mat_index].data;
    5420:	f9404be0 	ldr	x0, [sp, #144]
    5424:	f9401002 	ldr	x2, [x0, #32]
    5428:	b9408fe1 	ldr	w1, [sp, #140]
    542c:	aa0103e0 	mov	x0, x1
    5430:	d37ff800 	lsl	x0, x0, #1
    5434:	8b010000 	add	x0, x0, x1
    5438:	d37df000 	lsl	x0, x0, #3
    543c:	8b000040 	add	x0, x2, x0
    5440:	f9400800 	ldr	x0, [x0, #16]
    5444:	f9003fe0 	str	x0, [sp, #120]
		sptIndex tmp_i = times_inds[x];
    5448:	f94077e0 	ldr	x0, [sp, #232]
    544c:	d37ef400 	lsl	x0, x0, #2
    5450:	f9403fe1 	ldr	x1, [sp, #120]
    5454:	8b000020 	add	x0, x1, x0
    5458:	b9400000 	ldr	w0, [x0]
    545c:	b90077e0 	str	w0, [sp, #116]
		sptValue const entry = vals[x];
    5460:	f94077e0 	ldr	x0, [sp, #232]
    5464:	d37ef400 	lsl	x0, x0, #2
    5468:	f9405fe1 	ldr	x1, [sp, #184]
    546c:	8b000020 	add	x0, x1, x0
    5470:	bd400000 	ldr	s0, [x0]
    5474:	bd0073e0 	str	s0, [sp, #112]
#pragma omp simd
    5478:	b940c7e3 	ldr	w3, [sp, #196]
    547c:	b900e7ff 	str	wzr, [sp, #228]
    5480:	b940e7e0 	ldr	w0, [sp, #228]
    5484:	6b03001f 	cmp	w0, w3
    5488:	54000f83 	b.cc	5678 <sptOmpMTTKRP._omp_fn.0+0x374>  // b.lo, b.ul, b.last
		for(sptIndex i=2; i<nmodes; ++i) {
    548c:	52800040 	mov	w0, #0x2                   	// #2
    5490:	b900e3e0 	str	w0, [sp, #224]
    5494:	b940e3e1 	ldr	w1, [sp, #224]
    5498:	b940afe0 	ldr	w0, [sp, #172]
    549c:	6b00003f 	cmp	w1, w0
    54a0:	540006e3 	b.cc	557c <sptOmpMTTKRP._omp_fn.0+0x278>  // b.lo, b.ul, b.last
		sptIndex const mode_i = mode_ind[x];
    54a4:	f94077e0 	ldr	x0, [sp, #232]
    54a8:	d37ef400 	lsl	x0, x0, #2
    54ac:	f94067e1 	ldr	x1, [sp, #200]
    54b0:	8b000020 	add	x0, x1, x0
    54b4:	b9400000 	ldr	w0, [x0]
    54b8:	b9006fe0 	str	w0, [sp, #108]
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    54bc:	b9406fe1 	ldr	w1, [sp, #108]
    54c0:	b940c3e0 	ldr	w0, [sp, #192]
    54c4:	1b007c20 	mul	w0, w1, w0
    54c8:	2a0003e0 	mov	w0, w0
    54cc:	d37ef400 	lsl	x0, x0, #2
    54d0:	f9406be1 	ldr	x1, [sp, #208]
    54d4:	8b000020 	add	x0, x1, x0
    54d8:	f90033e0 	str	x0, [sp, #96]
		for(sptIndex r=0; r<R; ++r) {
    54dc:	b900dfff 	str	wzr, [sp, #220]
    54e0:	b940dfe1 	ldr	w1, [sp, #220]
    54e4:	b940c7e0 	ldr	w0, [sp, #196]
    54e8:	6b00003f 	cmp	w1, w0
    54ec:	54000143 	b.cc	5514 <sptOmpMTTKRP._omp_fn.0+0x210>  // b.lo, b.ul, b.last
		sptFreeValueVector(&scratch);
    54f0:	910123e0 	add	x0, sp, #0x48
    54f4:	97fff462 	bl	267c <sptFreeValueVector>
    54f8:	f94077e0 	ldr	x0, [sp, #232]
    54fc:	91000400 	add	x0, x0, #0x1
    5500:	f90077e0 	str	x0, [sp, #232]
    5504:	f94077e0 	ldr	x0, [sp, #232]
    5508:	eb14001f 	cmp	x0, x20
    550c:	54fff6a3 	b.cc	53e0 <sptOmpMTTKRP._omp_fn.0+0xdc>  // b.lo, b.ul, b.last
#pragma omp parallel for schedule(static) num_threads(tk)
    5510:	14000073 	b	56dc <sptOmpMTTKRP._omp_fn.0+0x3d8>
			mvals_row[r] += scratch.data[r];
    5514:	f9402fe1 	ldr	x1, [sp, #88]
    5518:	b940dfe0 	ldr	w0, [sp, #220]
    551c:	d37ef400 	lsl	x0, x0, #2
    5520:	8b000020 	add	x0, x1, x0
    5524:	bd400008 	ldr	s8, [x0]
    5528:	b940dfe0 	ldr	w0, [sp, #220]
    552c:	d37ef400 	lsl	x0, x0, #2
    5530:	f94033e1 	ldr	x1, [sp, #96]
    5534:	8b000035 	add	x21, x1, x0
    5538:	b94002b3 	ldr	w19, [x21]
    553c:	1e270260 	fmov	s0, w19
    5540:	1e282800 	fadd	s0, s0, s8
    5544:	1e260000 	fmov	w0, s0
    5548:	aa1503e2 	mov	x2, x21
    554c:	2a0003e1 	mov	w1, w0
    5550:	2a1303e0 	mov	w0, w19
    5554:	9400035b 	bl	62c0 <__aarch64_cas4_acq_rel>
    5558:	6b13001f 	cmp	w0, w19
    555c:	2a1303e1 	mov	w1, w19
    5560:	2a0003f3 	mov	w19, w0
    5564:	6b01001f 	cmp	w0, w1
    5568:	54fffea1 	b.ne	553c <sptOmpMTTKRP._omp_fn.0+0x238>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    556c:	b940dfe0 	ldr	w0, [sp, #220]
    5570:	11000400 	add	w0, w0, #0x1
    5574:	b900dfe0 	str	w0, [sp, #220]
    5578:	17ffffda 	b	54e0 <sptOmpMTTKRP._omp_fn.0+0x1dc>
			times_mat_index = mats_order[i];
    557c:	b940e3e0 	ldr	w0, [sp, #224]
    5580:	d37ef400 	lsl	x0, x0, #2
    5584:	f94053e1 	ldr	x1, [sp, #160]
    5588:	8b000020 	add	x0, x1, x0
    558c:	b9400000 	ldr	w0, [x0]
    5590:	b9008fe0 	str	w0, [sp, #140]
			times_mat = mats[times_mat_index];
    5594:	b9408fe0 	ldr	w0, [sp, #140]
    5598:	d37df000 	lsl	x0, x0, #3
    559c:	f9404fe1 	ldr	x1, [sp, #152]
    55a0:	8b000020 	add	x0, x1, x0
    55a4:	f9400000 	ldr	x0, [x0]
    55a8:	f90043e0 	str	x0, [sp, #128]
			times_inds = X->inds[times_mat_index].data;
    55ac:	f9404be0 	ldr	x0, [sp, #144]
    55b0:	f9401002 	ldr	x2, [x0, #32]
    55b4:	b9408fe1 	ldr	w1, [sp, #140]
    55b8:	aa0103e0 	mov	x0, x1
    55bc:	d37ff800 	lsl	x0, x0, #1
    55c0:	8b010000 	add	x0, x0, x1
    55c4:	d37df000 	lsl	x0, x0, #3
    55c8:	8b000040 	add	x0, x2, x0
    55cc:	f9400800 	ldr	x0, [x0, #16]
    55d0:	f9003fe0 	str	x0, [sp, #120]
			tmp_i = times_inds[x];
    55d4:	f94077e0 	ldr	x0, [sp, #232]
    55d8:	d37ef400 	lsl	x0, x0, #2
    55dc:	f9403fe1 	ldr	x1, [sp, #120]
    55e0:	8b000020 	add	x0, x1, x0
    55e4:	b9400000 	ldr	w0, [x0]
    55e8:	b90077e0 	str	w0, [sp, #116]
#pragma omp simd
    55ec:	b940c7e3 	ldr	w3, [sp, #196]
    55f0:	b900dbff 	str	wzr, [sp, #216]
    55f4:	b940dbe0 	ldr	w0, [sp, #216]
    55f8:	6b03001f 	cmp	w0, w3
    55fc:	540000a3 	b.cc	5610 <sptOmpMTTKRP._omp_fn.0+0x30c>  // b.lo, b.ul, b.last
		for(sptIndex i=2; i<nmodes; ++i) {
    5600:	b940e3e0 	ldr	w0, [sp, #224]
    5604:	11000400 	add	w0, w0, #0x1
    5608:	b900e3e0 	str	w0, [sp, #224]
    560c:	17ffffa2 	b	5494 <sptOmpMTTKRP._omp_fn.0+0x190>
				scratch.data[r] *= times_mat->values[tmp_i * stride + r];
    5610:	f9402fe1 	ldr	x1, [sp, #88]
    5614:	b940dbe0 	ldr	w0, [sp, #216]
    5618:	d37ef400 	lsl	x0, x0, #2
    561c:	8b000020 	add	x0, x1, x0
    5620:	bd400001 	ldr	s1, [x0]
    5624:	f94043e0 	ldr	x0, [sp, #128]
    5628:	f9400801 	ldr	x1, [x0, #16]
    562c:	b94077e2 	ldr	w2, [sp, #116]
    5630:	b940c3e0 	ldr	w0, [sp, #192]
    5634:	1b007c42 	mul	w2, w2, w0
    5638:	b940dbe0 	ldr	w0, [sp, #216]
    563c:	0b000040 	add	w0, w2, w0
    5640:	2a0003e0 	mov	w0, w0
    5644:	d37ef400 	lsl	x0, x0, #2
    5648:	8b000020 	add	x0, x1, x0
    564c:	bd400000 	ldr	s0, [x0]
    5650:	f9402fe1 	ldr	x1, [sp, #88]
    5654:	b940dbe0 	ldr	w0, [sp, #216]
    5658:	d37ef400 	lsl	x0, x0, #2
    565c:	8b000020 	add	x0, x1, x0
    5660:	1e200820 	fmul	s0, s1, s0
    5664:	bd000000 	str	s0, [x0]
    5668:	b940dbe0 	ldr	w0, [sp, #216]
    566c:	11000400 	add	w0, w0, #0x1
    5670:	b900dbe0 	str	w0, [sp, #216]
    5674:	17ffffe0 	b	55f4 <sptOmpMTTKRP._omp_fn.0+0x2f0>
			scratch.data[r] = entry * times_mat->values[tmp_i * stride + r];
    5678:	f94043e0 	ldr	x0, [sp, #128]
    567c:	f9400801 	ldr	x1, [x0, #16]
    5680:	b94077e2 	ldr	w2, [sp, #116]
    5684:	b940c3e0 	ldr	w0, [sp, #192]
    5688:	1b007c42 	mul	w2, w2, w0
    568c:	b940e7e0 	ldr	w0, [sp, #228]
    5690:	0b000040 	add	w0, w2, w0
    5694:	2a0003e0 	mov	w0, w0
    5698:	d37ef400 	lsl	x0, x0, #2
    569c:	8b000020 	add	x0, x1, x0
    56a0:	bd400001 	ldr	s1, [x0]
    56a4:	f9402fe1 	ldr	x1, [sp, #88]
    56a8:	b940e7e0 	ldr	w0, [sp, #228]
    56ac:	d37ef400 	lsl	x0, x0, #2
    56b0:	8b000020 	add	x0, x1, x0
    56b4:	bd4073e0 	ldr	s0, [sp, #112]
    56b8:	1e200820 	fmul	s0, s1, s0
    56bc:	bd000000 	str	s0, [x0]
    56c0:	b940e7e0 	ldr	w0, [sp, #228]
    56c4:	11000400 	add	w0, w0, #0x1
    56c8:	b900e7e0 	str	w0, [sp, #228]
    56cc:	17ffff6d 	b	5480 <sptOmpMTTKRP._omp_fn.0+0x17c>
    56d0:	d2800001 	mov	x1, #0x0                   	// #0
    56d4:	91000400 	add	x0, x0, #0x1
    56d8:	17ffff3c 	b	53c8 <sptOmpMTTKRP._omp_fn.0+0xc4>
#pragma omp parallel for schedule(static) num_threads(tk)
    56dc:	d503201f 	nop
    56e0:	fd4017e8 	ldr	d8, [sp, #40]
    56e4:	a94153f3 	ldp	x19, x20, [sp, #16]
    56e8:	f94013f5 	ldr	x21, [sp, #32]
    56ec:	a8cf7bfd 	ldp	x29, x30, [sp], #240
    56f0:	d65f03c0 	ret

00000000000056f4 <sptOmpMTTKRP_3D._omp_fn.0>:
#pragma omp parallel for schedule(static) num_threads(tk)
    56f4:	a9b47bfd 	stp	x29, x30, [sp, #-192]!
    56f8:	910003fd 	mov	x29, sp
    56fc:	a90153f3 	stp	x19, x20, [sp, #16]
    5700:	f90013f5 	str	x21, [sp, #32]
    5704:	fd0017e8 	str	d8, [sp, #40]
    5708:	f9001fe0 	str	x0, [sp, #56]
    570c:	f9401fe0 	ldr	x0, [sp, #56]
    5710:	f9401c00 	ldr	x0, [x0, #56]
    5714:	f90057e0 	str	x0, [sp, #168]
    5718:	f9401fe0 	ldr	x0, [sp, #56]
    571c:	f9401800 	ldr	x0, [x0, #48]
    5720:	f90053e0 	str	x0, [sp, #160]
    5724:	f9401fe0 	ldr	x0, [sp, #56]
    5728:	f9401400 	ldr	x0, [x0, #40]
    572c:	f9004fe0 	str	x0, [sp, #152]
    5730:	f9401fe0 	ldr	x0, [sp, #56]
    5734:	f9401000 	ldr	x0, [x0, #32]
    5738:	f9004be0 	str	x0, [sp, #144]
    573c:	f9401fe0 	ldr	x0, [sp, #56]
    5740:	f9400c00 	ldr	x0, [x0, #24]
    5744:	f90047e0 	str	x0, [sp, #136]
    5748:	f9401fe0 	ldr	x0, [sp, #56]
    574c:	f9400800 	ldr	x0, [x0, #16]
    5750:	f90043e0 	str	x0, [sp, #128]
    5754:	f9401fe0 	ldr	x0, [sp, #56]
    5758:	b9404400 	ldr	w0, [x0, #68]
    575c:	b9007fe0 	str	w0, [sp, #124]
    5760:	f9401fe0 	ldr	x0, [sp, #56]
    5764:	b9404000 	ldr	w0, [x0, #64]
    5768:	b9007be0 	str	w0, [sp, #120]
    576c:	f9401fe0 	ldr	x0, [sp, #56]
    5770:	f9400400 	ldr	x0, [x0, #8]
    5774:	f9003be0 	str	x0, [sp, #112]
    5778:	f9401fe0 	ldr	x0, [sp, #56]
    577c:	f9400000 	ldr	x0, [x0]
    5780:	f90037e0 	str	x0, [sp, #104]
    5784:	f94037f3 	ldr	x19, [sp, #104]
    5788:	f100027f 	cmp	x19, #0x0
    578c:	54000dc0 	b.eq	5944 <sptOmpMTTKRP_3D._omp_fn.0+0x250>  // b.none
    5790:	97ffeec8 	bl	12b0 <omp_get_num_threads@plt>
    5794:	93407c14 	sxtw	x20, w0
    5798:	97ffee92 	bl	11e0 <omp_get_thread_num@plt>
    579c:	93407c02 	sxtw	x2, w0
    57a0:	9ad40a60 	udiv	x0, x19, x20
    57a4:	9ad40a61 	udiv	x1, x19, x20
    57a8:	9b147c21 	mul	x1, x1, x20
    57ac:	cb010261 	sub	x1, x19, x1
    57b0:	eb01005f 	cmp	x2, x1
    57b4:	54000c23 	b.cc	5938 <sptOmpMTTKRP_3D._omp_fn.0+0x244>  // b.lo, b.ul, b.last
    57b8:	9b027c02 	mul	x2, x0, x2
    57bc:	8b010041 	add	x1, x2, x1
    57c0:	8b000034 	add	x20, x1, x0
    57c4:	eb14003f 	cmp	x1, x20
    57c8:	54000be2 	b.cs	5944 <sptOmpMTTKRP_3D._omp_fn.0+0x250>  // b.hs, b.nlast
    57cc:	f9005fe1 	str	x1, [sp, #184]
		sptIndex mode_i = mode_ind[x];
    57d0:	f9405fe0 	ldr	x0, [sp, #184]
    57d4:	d37ef400 	lsl	x0, x0, #2
    57d8:	f94043e1 	ldr	x1, [sp, #128]
    57dc:	8b000020 	add	x0, x1, x0
    57e0:	b9400000 	ldr	w0, [x0]
    57e4:	b90067e0 	str	w0, [sp, #100]
		sptValue * const restrict mvals_row = mvals + mode_i * stride;
    57e8:	b94067e1 	ldr	w1, [sp, #100]
    57ec:	b9407be0 	ldr	w0, [sp, #120]
    57f0:	1b007c20 	mul	w0, w1, w0
    57f4:	2a0003e0 	mov	w0, w0
    57f8:	d37ef400 	lsl	x0, x0, #2
    57fc:	f94047e1 	ldr	x1, [sp, #136]
    5800:	8b000020 	add	x0, x1, x0
    5804:	f9002fe0 	str	x0, [sp, #88]
		sptIndex tmp_i_1 = times_inds_1[x];
    5808:	f9405fe0 	ldr	x0, [sp, #184]
    580c:	d37ef400 	lsl	x0, x0, #2
    5810:	f9404fe1 	ldr	x1, [sp, #152]
    5814:	8b000020 	add	x0, x1, x0
    5818:	b9400000 	ldr	w0, [x0]
    581c:	b90057e0 	str	w0, [sp, #84]
		sptIndex tmp_i_2 = times_inds_2[x];
    5820:	f9405fe0 	ldr	x0, [sp, #184]
    5824:	d37ef400 	lsl	x0, x0, #2
    5828:	f94057e1 	ldr	x1, [sp, #168]
    582c:	8b000020 	add	x0, x1, x0
    5830:	b9400000 	ldr	w0, [x0]
    5834:	b90053e0 	str	w0, [sp, #80]
		sptValue entry = vals[x];
    5838:	f9405fe0 	ldr	x0, [sp, #184]
    583c:	d37ef400 	lsl	x0, x0, #2
    5840:	f9403be1 	ldr	x1, [sp, #112]
    5844:	8b000020 	add	x0, x1, x0
    5848:	bd400000 	ldr	s0, [x0]
    584c:	bd004fe0 	str	s0, [sp, #76]
		for(sptIndex r=0; r<R; ++r) {
    5850:	b900b7ff 	str	wzr, [sp, #180]
    5854:	b940b7e1 	ldr	w1, [sp, #180]
    5858:	b9407fe0 	ldr	w0, [sp, #124]
    585c:	6b00003f 	cmp	w1, w0
    5860:	54000103 	b.cc	5880 <sptOmpMTTKRP_3D._omp_fn.0+0x18c>  // b.lo, b.ul, b.last
    5864:	f9405fe0 	ldr	x0, [sp, #184]
    5868:	91000400 	add	x0, x0, #0x1
    586c:	f9005fe0 	str	x0, [sp, #184]
    5870:	f9405fe0 	ldr	x0, [sp, #184]
    5874:	eb14001f 	cmp	x0, x20
    5878:	54fffac3 	b.cc	57d0 <sptOmpMTTKRP_3D._omp_fn.0+0xdc>  // b.lo, b.ul, b.last
#pragma omp parallel for schedule(static) num_threads(tk)
    587c:	14000032 	b	5944 <sptOmpMTTKRP_3D._omp_fn.0+0x250>
			mvals_row[r] += entry * times_mat_1->values[tmp_i_1 * stride + r] * times_mat_2->values[tmp_i_2 * stride + r];
    5880:	f94053e0 	ldr	x0, [sp, #160]
    5884:	f9400801 	ldr	x1, [x0, #16]
    5888:	b94053e2 	ldr	w2, [sp, #80]
    588c:	b9407be0 	ldr	w0, [sp, #120]
    5890:	1b007c42 	mul	w2, w2, w0
    5894:	b940b7e0 	ldr	w0, [sp, #180]
    5898:	0b000040 	add	w0, w2, w0
    589c:	2a0003e0 	mov	w0, w0
    58a0:	d37ef400 	lsl	x0, x0, #2
    58a4:	8b000020 	add	x0, x1, x0
    58a8:	bd400000 	ldr	s0, [x0]
    58ac:	f9404be0 	ldr	x0, [sp, #144]
    58b0:	f9400801 	ldr	x1, [x0, #16]
    58b4:	b94057e2 	ldr	w2, [sp, #84]
    58b8:	b9407be0 	ldr	w0, [sp, #120]
    58bc:	1b007c42 	mul	w2, w2, w0
    58c0:	b940b7e0 	ldr	w0, [sp, #180]
    58c4:	0b000040 	add	w0, w2, w0
    58c8:	2a0003e0 	mov	w0, w0
    58cc:	d37ef400 	lsl	x0, x0, #2
    58d0:	8b000020 	add	x0, x1, x0
    58d4:	bd400002 	ldr	s2, [x0]
    58d8:	bd404fe1 	ldr	s1, [sp, #76]
    58dc:	1e210841 	fmul	s1, s2, s1
    58e0:	1e200828 	fmul	s8, s1, s0
    58e4:	b940b7e0 	ldr	w0, [sp, #180]
    58e8:	d37ef400 	lsl	x0, x0, #2
    58ec:	f9402fe1 	ldr	x1, [sp, #88]
    58f0:	8b000035 	add	x21, x1, x0
    58f4:	b94002b3 	ldr	w19, [x21]
    58f8:	1e270260 	fmov	s0, w19
    58fc:	1e282800 	fadd	s0, s0, s8
    5900:	1e260000 	fmov	w0, s0
    5904:	aa1503e2 	mov	x2, x21
    5908:	2a0003e1 	mov	w1, w0
    590c:	2a1303e0 	mov	w0, w19
    5910:	9400026c 	bl	62c0 <__aarch64_cas4_acq_rel>
    5914:	6b13001f 	cmp	w0, w19
    5918:	2a1303e1 	mov	w1, w19
    591c:	2a0003f3 	mov	w19, w0
    5920:	6b01001f 	cmp	w0, w1
    5924:	54fffea1 	b.ne	58f8 <sptOmpMTTKRP_3D._omp_fn.0+0x204>  // b.any
		for(sptIndex r=0; r<R; ++r) {
    5928:	b940b7e0 	ldr	w0, [sp, #180]
    592c:	11000400 	add	w0, w0, #0x1
    5930:	b900b7e0 	str	w0, [sp, #180]
    5934:	17ffffc8 	b	5854 <sptOmpMTTKRP_3D._omp_fn.0+0x160>
    5938:	d2800001 	mov	x1, #0x0                   	// #0
    593c:	91000400 	add	x0, x0, #0x1
    5940:	17ffff9e 	b	57b8 <sptOmpMTTKRP_3D._omp_fn.0+0xc4>
#pragma omp parallel for schedule(static) num_threads(tk)
    5944:	d503201f 	nop
    5948:	fd4017e8 	ldr	d8, [sp, #40]
    594c:	a94153f3 	ldp	x19, x20, [sp, #16]
    5950:	f94013f5 	ldr	x21, [sp, #32]
    5954:	a8cc7bfd 	ldp	x29, x30, [sp], #192
    5958:	d65f03c0 	ret

000000000000595c <sptNewTimer>:
		int use_cuda;
		struct timespec start_timespec;
		struct timespec stop_timespec;
};

int sptNewTimer(sptTimer *timer, int use_cuda) {
    595c:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    5960:	910003fd 	mov	x29, sp
    5964:	f9000fe0 	str	x0, [sp, #24]
    5968:	b90017e1 	str	w1, [sp, #20]
	*timer = (sptTimer) malloc(sizeof **timer);
    596c:	d2800500 	mov	x0, #0x28                  	// #40
    5970:	97ffee2c 	bl	1220 <malloc@plt>
    5974:	aa0003e1 	mov	x1, x0
    5978:	f9400fe0 	ldr	x0, [sp, #24]
    597c:	f9000001 	str	x1, [x0]
	(*timer)->use_cuda = use_cuda;
    5980:	f9400fe0 	ldr	x0, [sp, #24]
    5984:	f9400000 	ldr	x0, [x0]
    5988:	b94017e1 	ldr	w1, [sp, #20]
    598c:	b9000001 	str	w1, [x0]
	if(use_cuda) {
    5990:	b94017e0 	ldr	w0, [sp, #20]
    5994:	7100001f 	cmp	w0, #0x0
    5998:	540001c0 	b.eq	59d0 <sptNewTimer+0x74>  // b.none
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    599c:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    59a0:	91326004 	add	x4, x0, #0xc98
    59a4:	528004e3 	mov	w3, #0x27                  	// #39
    59a8:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    59ac:	91330002 	add	x2, x0, #0xcc0
    59b0:	52800061 	mov	w1, #0x3                   	// #3
    59b4:	72a00041 	movk	w1, #0x2, lsl #16
    59b8:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    59bc:	9133a000 	add	x0, x0, #0xce8
    59c0:	9400016e 	bl	5f78 <spt_ComplainError>
    59c4:	52800060 	mov	w0, #0x3                   	// #3
    59c8:	72a00040 	movk	w0, #0x2, lsl #16
    59cc:	14000002 	b	59d4 <sptNewTimer+0x78>
	}
	return 0;
    59d0:	52800000 	mov	w0, #0x0                   	// #0
}
    59d4:	a8c27bfd 	ldp	x29, x30, [sp], #32
    59d8:	d65f03c0 	ret

00000000000059dc <sptStartTimer>:

int sptStartTimer(sptTimer timer) {
    59dc:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    59e0:	910003fd 	mov	x29, sp
    59e4:	f9000bf3 	str	x19, [sp, #16]
    59e8:	f90017e0 	str	x0, [sp, #40]
	int result;
	if(timer->use_cuda) {
    59ec:	f94017e0 	ldr	x0, [sp, #40]
    59f0:	b9400000 	ldr	w0, [x0]
    59f4:	7100001f 	cmp	w0, #0x0
    59f8:	540001c0 	b.eq	5a30 <sptStartTimer+0x54>  // b.none
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    59fc:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5a00:	91326004 	add	x4, x0, #0xc98
    5a04:	528005e3 	mov	w3, #0x2f                  	// #47
    5a08:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5a0c:	91330002 	add	x2, x0, #0xcc0
    5a10:	52800061 	mov	w1, #0x3                   	// #3
    5a14:	72a00041 	movk	w1, #0x2, lsl #16
    5a18:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5a1c:	9133a000 	add	x0, x0, #0xce8
    5a20:	94000156 	bl	5f78 <spt_ComplainError>
    5a24:	52800060 	mov	w0, #0x3                   	// #3
    5a28:	72a00040 	movk	w0, #0x2, lsl #16
    5a2c:	14000029 	b	5ad0 <sptStartTimer+0xf4>
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->start_timespec);
    5a30:	f94017e0 	ldr	x0, [sp, #40]
    5a34:	91002000 	add	x0, x0, #0x8
    5a38:	aa0003e1 	mov	x1, x0
    5a3c:	52800020 	mov	w0, #0x1                   	// #1
    5a40:	97ffeddc 	bl	11b0 <clock_gettime@plt>
    5a44:	b9003fe0 	str	w0, [sp, #60]
		spt_CheckOSError(result, "Timer New");
    5a48:	b9403fe0 	ldr	w0, [sp, #60]
    5a4c:	7100001f 	cmp	w0, #0x0
    5a50:	1a9f07e0 	cset	w0, ne  // ne = any
    5a54:	12001c00 	and	w0, w0, #0xff
    5a58:	92401c00 	and	x0, x0, #0xff
    5a5c:	f100001f 	cmp	x0, #0x0
    5a60:	54000360 	b.eq	5acc <sptStartTimer+0xf0>  // b.none
    5a64:	97ffee57 	bl	13c0 <__errno_location@plt>
    5a68:	b9400000 	ldr	w0, [x0]
    5a6c:	3140401f 	cmn	w0, #0x10, lsl #12
    5a70:	1a9f07e0 	cset	w0, ne  // ne = any
    5a74:	12001c00 	and	w0, w0, #0xff
    5a78:	92401c00 	and	x0, x0, #0xff
    5a7c:	f100001f 	cmp	x0, #0x0
    5a80:	54000260 	b.eq	5acc <sptStartTimer+0xf0>  // b.none
    5a84:	97ffee4f 	bl	13c0 <__errno_location@plt>
    5a88:	b9400000 	ldr	w0, [x0]
    5a8c:	11404013 	add	w19, w0, #0x10, lsl #12
    5a90:	97ffee4c 	bl	13c0 <__errno_location@plt>
    5a94:	b9400000 	ldr	w0, [x0]
    5a98:	97ffee0a 	bl	12c0 <strerror@plt>
    5a9c:	aa0003e4 	mov	x4, x0
    5aa0:	52800643 	mov	w3, #0x32                  	// #50
    5aa4:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5aa8:	91330002 	add	x2, x0, #0xcc0
    5aac:	2a1303e1 	mov	w1, w19
    5ab0:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5ab4:	9133a000 	add	x0, x0, #0xce8
    5ab8:	94000130 	bl	5f78 <spt_ComplainError>
    5abc:	97ffee41 	bl	13c0 <__errno_location@plt>
    5ac0:	b9400000 	ldr	w0, [x0]
    5ac4:	11404000 	add	w0, w0, #0x10, lsl #12
    5ac8:	14000002 	b	5ad0 <sptStartTimer+0xf4>
	}
	return 0;
    5acc:	52800000 	mov	w0, #0x0                   	// #0
}
    5ad0:	f9400bf3 	ldr	x19, [sp, #16]
    5ad4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    5ad8:	d65f03c0 	ret

0000000000005adc <sptStopTimer>:

int sptStopTimer(sptTimer timer) {
    5adc:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    5ae0:	910003fd 	mov	x29, sp
    5ae4:	f9000bf3 	str	x19, [sp, #16]
    5ae8:	f90017e0 	str	x0, [sp, #40]
	int result;
	if(timer->use_cuda) {
    5aec:	f94017e0 	ldr	x0, [sp, #40]
    5af0:	b9400000 	ldr	w0, [x0]
    5af4:	7100001f 	cmp	w0, #0x0
    5af8:	540001c0 	b.eq	5b30 <sptStopTimer+0x54>  // b.none
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    5afc:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5b00:	91326004 	add	x4, x0, #0xc98
    5b04:	52800743 	mov	w3, #0x3a                  	// #58
    5b08:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5b0c:	91330002 	add	x2, x0, #0xcc0
    5b10:	52800061 	mov	w1, #0x3                   	// #3
    5b14:	72a00041 	movk	w1, #0x2, lsl #16
    5b18:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5b1c:	9133a000 	add	x0, x0, #0xce8
    5b20:	94000116 	bl	5f78 <spt_ComplainError>
    5b24:	52800060 	mov	w0, #0x3                   	// #3
    5b28:	72a00040 	movk	w0, #0x2, lsl #16
    5b2c:	14000029 	b	5bd0 <sptStopTimer+0xf4>
	} else {
		result = clock_gettime(CLOCK_MONOTONIC, &timer->stop_timespec);
    5b30:	f94017e0 	ldr	x0, [sp, #40]
    5b34:	91006000 	add	x0, x0, #0x18
    5b38:	aa0003e1 	mov	x1, x0
    5b3c:	52800020 	mov	w0, #0x1                   	// #1
    5b40:	97ffed9c 	bl	11b0 <clock_gettime@plt>
    5b44:	b9003fe0 	str	w0, [sp, #60]
		spt_CheckOSError(result, "Timer New");
    5b48:	b9403fe0 	ldr	w0, [sp, #60]
    5b4c:	7100001f 	cmp	w0, #0x0
    5b50:	1a9f07e0 	cset	w0, ne  // ne = any
    5b54:	12001c00 	and	w0, w0, #0xff
    5b58:	92401c00 	and	x0, x0, #0xff
    5b5c:	f100001f 	cmp	x0, #0x0
    5b60:	54000360 	b.eq	5bcc <sptStopTimer+0xf0>  // b.none
    5b64:	97ffee17 	bl	13c0 <__errno_location@plt>
    5b68:	b9400000 	ldr	w0, [x0]
    5b6c:	3140401f 	cmn	w0, #0x10, lsl #12
    5b70:	1a9f07e0 	cset	w0, ne  // ne = any
    5b74:	12001c00 	and	w0, w0, #0xff
    5b78:	92401c00 	and	x0, x0, #0xff
    5b7c:	f100001f 	cmp	x0, #0x0
    5b80:	54000260 	b.eq	5bcc <sptStopTimer+0xf0>  // b.none
    5b84:	97ffee0f 	bl	13c0 <__errno_location@plt>
    5b88:	b9400000 	ldr	w0, [x0]
    5b8c:	11404013 	add	w19, w0, #0x10, lsl #12
    5b90:	97ffee0c 	bl	13c0 <__errno_location@plt>
    5b94:	b9400000 	ldr	w0, [x0]
    5b98:	97ffedca 	bl	12c0 <strerror@plt>
    5b9c:	aa0003e4 	mov	x4, x0
    5ba0:	528007a3 	mov	w3, #0x3d                  	// #61
    5ba4:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5ba8:	91330002 	add	x2, x0, #0xcc0
    5bac:	2a1303e1 	mov	w1, w19
    5bb0:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5bb4:	9133a000 	add	x0, x0, #0xce8
    5bb8:	940000f0 	bl	5f78 <spt_ComplainError>
    5bbc:	97ffee01 	bl	13c0 <__errno_location@plt>
    5bc0:	b9400000 	ldr	w0, [x0]
    5bc4:	11404000 	add	w0, w0, #0x10, lsl #12
    5bc8:	14000002 	b	5bd0 <sptStopTimer+0xf4>
	}
	return 0;
    5bcc:	52800000 	mov	w0, #0x0                   	// #0
}
    5bd0:	f9400bf3 	ldr	x19, [sp, #16]
    5bd4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    5bd8:	d65f03c0 	ret

0000000000005bdc <sptElapsedTime>:

double sptElapsedTime(const sptTimer timer) {
    5bdc:	d10043ff 	sub	sp, sp, #0x10
    5be0:	f90007e0 	str	x0, [sp, #8]
	if(timer->use_cuda) {
    5be4:	f94007e0 	ldr	x0, [sp, #8]
    5be8:	b9400000 	ldr	w0, [x0]
    5bec:	7100001f 	cmp	w0, #0x0
    5bf0:	54000080 	b.eq	5c00 <sptElapsedTime+0x24>  // b.none
		return NAN;
    5bf4:	d2efff00 	mov	x0, #0x7ff8000000000000    	// #9221120237041090560
    5bf8:	9e670000 	fmov	d0, x0
    5bfc:	14000013 	b	5c48 <sptElapsedTime+0x6c>
	} else {
		return timer->stop_timespec.tv_sec - timer->start_timespec.tv_sec
    5c00:	f94007e0 	ldr	x0, [sp, #8]
    5c04:	f9400c01 	ldr	x1, [x0, #24]
    5c08:	f94007e0 	ldr	x0, [sp, #8]
    5c0c:	f9400400 	ldr	x0, [x0, #8]
    5c10:	cb000020 	sub	x0, x1, x0
    5c14:	9e670000 	fmov	d0, x0
					 + (timer->stop_timespec.tv_nsec - timer->start_timespec.tv_nsec) * 1e-9;
    5c18:	5e61d801 	scvtf	d1, d0
    5c1c:	f94007e0 	ldr	x0, [sp, #8]
    5c20:	f9401001 	ldr	x1, [x0, #32]
    5c24:	f94007e0 	ldr	x0, [sp, #8]
    5c28:	f9400800 	ldr	x0, [x0, #16]
    5c2c:	cb000020 	sub	x0, x1, x0
    5c30:	9e670000 	fmov	d0, x0
    5c34:	5e61d800 	scvtf	d0, d0
    5c38:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5c3c:	fd468402 	ldr	d2, [x0, #3336]
    5c40:	1e620800 	fmul	d0, d0, d2
    5c44:	1e602820 	fadd	d0, d1, d0
	}
}
    5c48:	910043ff 	add	sp, sp, #0x10
    5c4c:	d65f03c0 	ret

0000000000005c50 <sptPrintElapsedTime>:

double sptPrintElapsedTime(const sptTimer timer, const char *name) {
    5c50:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    5c54:	910003fd 	mov	x29, sp
    5c58:	f9000fe0 	str	x0, [sp, #24]
    5c5c:	f9000be1 	str	x1, [sp, #16]
	double elapsed_time = sptElapsedTime(timer);
    5c60:	f9400fe0 	ldr	x0, [sp, #24]
    5c64:	97ffffde 	bl	5bdc <sptElapsedTime>
    5c68:	fd0017e0 	str	d0, [sp, #40]
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    5c6c:	d0000080 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    5c70:	f947e000 	ldr	x0, [x0, #4032]
    5c74:	f9400003 	ldr	x3, [x0]
    5c78:	fd4017e0 	ldr	d0, [sp, #40]
    5c7c:	f9400be2 	ldr	x2, [sp, #16]
    5c80:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5c84:	9133e001 	add	x1, x0, #0xcf8
    5c88:	aa0303e0 	mov	x0, x3
    5c8c:	97ffedd9 	bl	13f0 <fprintf@plt>
	return elapsed_time;
    5c90:	fd4017e0 	ldr	d0, [sp, #40]
}
    5c94:	a8c37bfd 	ldp	x29, x30, [sp], #48
    5c98:	d65f03c0 	ret

0000000000005c9c <sptPrintAverageElapsedTime>:


double sptPrintAverageElapsedTime(const sptTimer timer, const int niters, const char *name) {
    5c9c:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    5ca0:	910003fd 	mov	x29, sp
    5ca4:	f90017e0 	str	x0, [sp, #40]
    5ca8:	b90027e1 	str	w1, [sp, #36]
    5cac:	f9000fe2 	str	x2, [sp, #24]
	double elapsed_time = sptElapsedTime(timer) / niters;
    5cb0:	f94017e0 	ldr	x0, [sp, #40]
    5cb4:	97ffffca 	bl	5bdc <sptElapsedTime>
    5cb8:	1e604001 	fmov	d1, d0
    5cbc:	b94027e0 	ldr	w0, [sp, #36]
    5cc0:	1e620000 	scvtf	d0, w0
    5cc4:	1e601820 	fdiv	d0, d1, d0
    5cc8:	fd001fe0 	str	d0, [sp, #56]
	fprintf(stdout, "[%s]: %.9lf s\n", name, elapsed_time);
    5ccc:	d0000080 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    5cd0:	f947e000 	ldr	x0, [x0, #4032]
    5cd4:	f9400003 	ldr	x3, [x0]
    5cd8:	fd401fe0 	ldr	d0, [sp, #56]
    5cdc:	f9400fe2 	ldr	x2, [sp, #24]
    5ce0:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5ce4:	9133e001 	add	x1, x0, #0xcf8
    5ce8:	aa0303e0 	mov	x0, x3
    5cec:	97ffedc1 	bl	13f0 <fprintf@plt>
	return elapsed_time;
    5cf0:	fd401fe0 	ldr	d0, [sp, #56]
}
    5cf4:	a8c47bfd 	ldp	x29, x30, [sp], #64
    5cf8:	d65f03c0 	ret

0000000000005cfc <sptFreeTimer>:


int sptFreeTimer(sptTimer timer) {
    5cfc:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    5d00:	910003fd 	mov	x29, sp
    5d04:	f9000fe0 	str	x0, [sp, #24]
	if(timer->use_cuda) {
    5d08:	f9400fe0 	ldr	x0, [sp, #24]
    5d0c:	b9400000 	ldr	w0, [x0]
    5d10:	7100001f 	cmp	w0, #0x0
    5d14:	540001c0 	b.eq	5d4c <sptFreeTimer+0x50>  // b.none
		spt_CheckError(3 + SPTERR_CUDA_ERROR, "Timer New", "CUDA support is disabled in this build");
    5d18:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5d1c:	91326004 	add	x4, x0, #0xc98
    5d20:	52800b63 	mov	w3, #0x5b                  	// #91
    5d24:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5d28:	91330002 	add	x2, x0, #0xcc0
    5d2c:	52800061 	mov	w1, #0x3                   	// #3
    5d30:	72a00041 	movk	w1, #0x2, lsl #16
    5d34:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5d38:	9133a000 	add	x0, x0, #0xce8
    5d3c:	9400008f 	bl	5f78 <spt_ComplainError>
    5d40:	52800060 	mov	w0, #0x3                   	// #3
    5d44:	72a00040 	movk	w0, #0x2, lsl #16
    5d48:	14000004 	b	5d58 <sptFreeTimer+0x5c>
	}
	free(timer);
    5d4c:	f9400fe0 	ldr	x0, [sp, #24]
    5d50:	97ffed80 	bl	1350 <free@plt>
	return 0;
    5d54:	52800000 	mov	w0, #0x0                   	// #0
}
    5d58:	a8c27bfd 	ldp	x29, x30, [sp], #32
    5d5c:	d65f03c0 	ret

0000000000005d60 <sptDumpMatrix>:
 *
 * @param mtx   a valid pointer to a sptMatrix variable
 * @param fp a file pointer
 *
 */
int sptDumpMatrix(sptMatrix *mtx, FILE *fp) {
    5d60:	a9bb7bfd 	stp	x29, x30, [sp, #-80]!
    5d64:	910003fd 	mov	x29, sp
    5d68:	f9000bf3 	str	x19, [sp, #16]
    5d6c:	f90017e0 	str	x0, [sp, #40]
    5d70:	f90013e1 	str	x1, [sp, #32]
	int iores;
	sptIndex nrows = mtx->nrows;
    5d74:	f94017e0 	ldr	x0, [sp, #40]
    5d78:	b9400000 	ldr	w0, [x0]
    5d7c:	b90047e0 	str	w0, [sp, #68]
	sptIndex ncols = mtx->ncols;
    5d80:	f94017e0 	ldr	x0, [sp, #40]
    5d84:	b9400400 	ldr	w0, [x0, #4]
    5d88:	b90043e0 	str	w0, [sp, #64]
	sptIndex stride = mtx->stride;
    5d8c:	f94017e0 	ldr	x0, [sp, #40]
    5d90:	b9400c00 	ldr	w0, [x0, #12]
    5d94:	b9003fe0 	str	w0, [sp, #60]
	iores = fprintf(fp, "%"PASTA_PRI_INDEX " x %"PASTA_PRI_INDEX " matrix\n", nrows, ncols);
    5d98:	b94043e3 	ldr	w3, [sp, #64]
    5d9c:	b94047e2 	ldr	w2, [sp, #68]
    5da0:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5da4:	91344001 	add	x1, x0, #0xd10
    5da8:	f94013e0 	ldr	x0, [sp, #32]
    5dac:	97ffed91 	bl	13f0 <fprintf@plt>
    5db0:	b9003be0 	str	w0, [sp, #56]
	spt_CheckOSError(iores < 0, "Mtx Dump");
    5db4:	b9403be0 	ldr	w0, [sp, #56]
    5db8:	531f7c00 	lsr	w0, w0, #31
    5dbc:	12001c00 	and	w0, w0, #0xff
    5dc0:	92401c00 	and	x0, x0, #0xff
    5dc4:	f100001f 	cmp	x0, #0x0
    5dc8:	54000360 	b.eq	5e34 <sptDumpMatrix+0xd4>  // b.none
    5dcc:	97ffed7d 	bl	13c0 <__errno_location@plt>
    5dd0:	b9400000 	ldr	w0, [x0]
    5dd4:	3140401f 	cmn	w0, #0x10, lsl #12
    5dd8:	1a9f07e0 	cset	w0, ne  // ne = any
    5ddc:	12001c00 	and	w0, w0, #0xff
    5de0:	92401c00 	and	x0, x0, #0xff
    5de4:	f100001f 	cmp	x0, #0x0
    5de8:	54000260 	b.eq	5e34 <sptDumpMatrix+0xd4>  // b.none
    5dec:	97ffed75 	bl	13c0 <__errno_location@plt>
    5df0:	b9400000 	ldr	w0, [x0]
    5df4:	11404013 	add	w19, w0, #0x10, lsl #12
    5df8:	97ffed72 	bl	13c0 <__errno_location@plt>
    5dfc:	b9400000 	ldr	w0, [x0]
    5e00:	97ffed30 	bl	12c0 <strerror@plt>
    5e04:	aa0003e4 	mov	x4, x0
    5e08:	52800543 	mov	w3, #0x2a                  	// #42
    5e0c:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5e10:	91348002 	add	x2, x0, #0xd20
    5e14:	2a1303e1 	mov	w1, w19
    5e18:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5e1c:	91352000 	add	x0, x0, #0xd48
    5e20:	94000056 	bl	5f78 <spt_ComplainError>
    5e24:	97ffed67 	bl	13c0 <__errno_location@plt>
    5e28:	b9400000 	ldr	w0, [x0]
    5e2c:	11404000 	add	w0, w0, #0x10, lsl #12
    5e30:	1400004f 	b	5f6c <sptDumpMatrix+0x20c>
	for(sptIndex i=0; i < nrows; ++i) {
    5e34:	b9004fff 	str	wzr, [sp, #76]
    5e38:	14000043 	b	5f44 <sptDumpMatrix+0x1e4>
		for(sptIndex j=0; j < ncols; ++j) {
    5e3c:	b9004bff 	str	wzr, [sp, #72]
    5e40:	14000035 	b	5f14 <sptDumpMatrix+0x1b4>
			iores = fprintf(fp, "%.2"PASTA_PRI_VALUE "\t", mtx->values[i * stride + j]);
    5e44:	f94017e0 	ldr	x0, [sp, #40]
    5e48:	f9400801 	ldr	x1, [x0, #16]
    5e4c:	b9404fe2 	ldr	w2, [sp, #76]
    5e50:	b9403fe0 	ldr	w0, [sp, #60]
    5e54:	1b007c42 	mul	w2, w2, w0
    5e58:	b9404be0 	ldr	w0, [sp, #72]
    5e5c:	0b000040 	add	w0, w2, w0
    5e60:	2a0003e0 	mov	w0, w0
    5e64:	d37ef400 	lsl	x0, x0, #2
    5e68:	8b000020 	add	x0, x1, x0
    5e6c:	bd400000 	ldr	s0, [x0]
    5e70:	1e22c000 	fcvt	d0, s0
    5e74:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5e78:	91356001 	add	x1, x0, #0xd58
    5e7c:	f94013e0 	ldr	x0, [sp, #32]
    5e80:	97ffed5c 	bl	13f0 <fprintf@plt>
    5e84:	b9003be0 	str	w0, [sp, #56]
			spt_CheckOSError(iores < 0, "Mtx Dump");
    5e88:	b9403be0 	ldr	w0, [sp, #56]
    5e8c:	531f7c00 	lsr	w0, w0, #31
    5e90:	12001c00 	and	w0, w0, #0xff
    5e94:	92401c00 	and	x0, x0, #0xff
    5e98:	f100001f 	cmp	x0, #0x0
    5e9c:	54000360 	b.eq	5f08 <sptDumpMatrix+0x1a8>  // b.none
    5ea0:	97ffed48 	bl	13c0 <__errno_location@plt>
    5ea4:	b9400000 	ldr	w0, [x0]
    5ea8:	3140401f 	cmn	w0, #0x10, lsl #12
    5eac:	1a9f07e0 	cset	w0, ne  // ne = any
    5eb0:	12001c00 	and	w0, w0, #0xff
    5eb4:	92401c00 	and	x0, x0, #0xff
    5eb8:	f100001f 	cmp	x0, #0x0
    5ebc:	54000260 	b.eq	5f08 <sptDumpMatrix+0x1a8>  // b.none
    5ec0:	97ffed40 	bl	13c0 <__errno_location@plt>
    5ec4:	b9400000 	ldr	w0, [x0]
    5ec8:	11404013 	add	w19, w0, #0x10, lsl #12
    5ecc:	97ffed3d 	bl	13c0 <__errno_location@plt>
    5ed0:	b9400000 	ldr	w0, [x0]
    5ed4:	97ffecfb 	bl	12c0 <strerror@plt>
    5ed8:	aa0003e4 	mov	x4, x0
    5edc:	528005c3 	mov	w3, #0x2e                  	// #46
    5ee0:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5ee4:	91348002 	add	x2, x0, #0xd20
    5ee8:	2a1303e1 	mov	w1, w19
    5eec:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5ef0:	91352000 	add	x0, x0, #0xd48
    5ef4:	94000021 	bl	5f78 <spt_ComplainError>
    5ef8:	97ffed32 	bl	13c0 <__errno_location@plt>
    5efc:	b9400000 	ldr	w0, [x0]
    5f00:	11404000 	add	w0, w0, #0x10, lsl #12
    5f04:	1400001a 	b	5f6c <sptDumpMatrix+0x20c>
		for(sptIndex j=0; j < ncols; ++j) {
    5f08:	b9404be0 	ldr	w0, [sp, #72]
    5f0c:	11000400 	add	w0, w0, #0x1
    5f10:	b9004be0 	str	w0, [sp, #72]
    5f14:	b9404be1 	ldr	w1, [sp, #72]
    5f18:	b94043e0 	ldr	w0, [sp, #64]
    5f1c:	6b00003f 	cmp	w1, w0
    5f20:	54fff923 	b.cc	5e44 <sptDumpMatrix+0xe4>  // b.lo, b.ul, b.last
		}
		iores = fprintf(fp, "\n");
    5f24:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5f28:	91358001 	add	x1, x0, #0xd60
    5f2c:	f94013e0 	ldr	x0, [sp, #32]
    5f30:	97ffed30 	bl	13f0 <fprintf@plt>
    5f34:	b9003be0 	str	w0, [sp, #56]
	for(sptIndex i=0; i < nrows; ++i) {
    5f38:	b9404fe0 	ldr	w0, [sp, #76]
    5f3c:	11000400 	add	w0, w0, #0x1
    5f40:	b9004fe0 	str	w0, [sp, #76]
    5f44:	b9404fe1 	ldr	w1, [sp, #76]
    5f48:	b94047e0 	ldr	w0, [sp, #68]
    5f4c:	6b00003f 	cmp	w1, w0
    5f50:	54fff763 	b.cc	5e3c <sptDumpMatrix+0xdc>  // b.lo, b.ul, b.last
	}
	iores = fprintf(fp, "\n");
    5f54:	b0000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    5f58:	91358001 	add	x1, x0, #0xd60
    5f5c:	f94013e0 	ldr	x0, [sp, #32]
    5f60:	97ffed24 	bl	13f0 <fprintf@plt>
    5f64:	b9003be0 	str	w0, [sp, #56]
	return 0;
    5f68:	52800000 	mov	w0, #0x0                   	// #0
}
    5f6c:	f9400bf3 	ldr	x19, [sp, #16]
    5f70:	a8c57bfd 	ldp	x29, x30, [sp], #80
    5f74:	d65f03c0 	ret

0000000000005f78 <spt_ComplainError>:

/**
 * Set last error information as specified and print the error message.
 * Should not be called directly, use the macro `spt_CheckError`.
 */
void spt_ComplainError(const char *module, int errcode, const char *file, unsigned line, const char *reason) {
    5f78:	a9bd7bfd 	stp	x29, x30, [sp, #-48]!
    5f7c:	910003fd 	mov	x29, sp
    5f80:	f90017e0 	str	x0, [sp, #40]
    5f84:	b90027e1 	str	w1, [sp, #36]
    5f88:	f9000fe2 	str	x2, [sp, #24]
    5f8c:	b90023e3 	str	w3, [sp, #32]
    5f90:	f9000be4 	str	x4, [sp, #16]
	g_last_error.errcode = errcode;
    5f94:	d53bd040 	mrs	x0, tpidr_el0
    5f98:	91400000 	add	x0, x0, #0x0, lsl #12
    5f9c:	91004000 	add	x0, x0, #0x10
    5fa0:	b94027e1 	ldr	w1, [sp, #36]
    5fa4:	b9000801 	str	w1, [x0, #8]
	g_last_error.module = module;
    5fa8:	d53bd040 	mrs	x0, tpidr_el0
    5fac:	91400000 	add	x0, x0, #0x0, lsl #12
    5fb0:	91004000 	add	x0, x0, #0x10
    5fb4:	f94017e1 	ldr	x1, [sp, #40]
    5fb8:	f9000001 	str	x1, [x0]
	g_last_error.file = file;
    5fbc:	d53bd040 	mrs	x0, tpidr_el0
    5fc0:	91400000 	add	x0, x0, #0x0, lsl #12
    5fc4:	91004000 	add	x0, x0, #0x10
    5fc8:	f9400fe1 	ldr	x1, [sp, #24]
    5fcc:	f9000801 	str	x1, [x0, #16]
	g_last_error.line = line;
    5fd0:	d53bd040 	mrs	x0, tpidr_el0
    5fd4:	91400000 	add	x0, x0, #0x0, lsl #12
    5fd8:	91004000 	add	x0, x0, #0x10
    5fdc:	b94023e1 	ldr	w1, [sp, #32]
    5fe0:	b9001801 	str	w1, [x0, #24]
	if(reason) {
    5fe4:	f9400be0 	ldr	x0, [sp, #16]
    5fe8:	f100001f 	cmp	x0, #0x0
    5fec:	54000280 	b.eq	603c <spt_ComplainError+0xc4>  // b.none
		free(g_last_error.reason);
    5ff0:	d53bd040 	mrs	x0, tpidr_el0
    5ff4:	91400000 	add	x0, x0, #0x0, lsl #12
    5ff8:	91004000 	add	x0, x0, #0x10
    5ffc:	f9401000 	ldr	x0, [x0, #32]
    6000:	97ffecd4 	bl	1350 <free@plt>
		g_last_error.reason = strdup(reason);
    6004:	f9400be0 	ldr	x0, [sp, #16]
    6008:	97ffeca6 	bl	12a0 <strdup@plt>
    600c:	aa0003e1 	mov	x1, x0
    6010:	d53bd040 	mrs	x0, tpidr_el0
    6014:	91400000 	add	x0, x0, #0x0, lsl #12
    6018:	91004000 	add	x0, x0, #0x10
    601c:	f9001001 	str	x1, [x0, #32]
		if(!g_last_error.reason) {
    6020:	d53bd040 	mrs	x0, tpidr_el0
    6024:	91400000 	add	x0, x0, #0x0, lsl #12
    6028:	91004000 	add	x0, x0, #0x10
    602c:	f9401000 	ldr	x0, [x0, #32]
    6030:	f100001f 	cmp	x0, #0x0
    6034:	54000041 	b.ne	603c <spt_ComplainError+0xc4>  // b.any
			abort();
    6038:	97ffecb2 	bl	1300 <abort@plt>
		}
	}
	if(g_last_error.reason && g_last_error.reason[0] != '\0') {
    603c:	d53bd040 	mrs	x0, tpidr_el0
    6040:	91400000 	add	x0, x0, #0x0, lsl #12
    6044:	91004000 	add	x0, x0, #0x10
    6048:	f9401000 	ldr	x0, [x0, #32]
    604c:	f100001f 	cmp	x0, #0x0
    6050:	54000520 	b.eq	60f4 <spt_ComplainError+0x17c>  // b.none
    6054:	d53bd040 	mrs	x0, tpidr_el0
    6058:	91400000 	add	x0, x0, #0x0, lsl #12
    605c:	91004000 	add	x0, x0, #0x10
    6060:	f9401000 	ldr	x0, [x0, #32]
    6064:	39400000 	ldrb	w0, [x0]
    6068:	7100001f 	cmp	w0, #0x0
    606c:	54000440 	b.eq	60f4 <spt_ComplainError+0x17c>  // b.none
		fprintf(stderr, "[%s] error 0x%08x at %s:%u, %s\n",
    6070:	b0000080 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    6074:	f947d800 	ldr	x0, [x0, #4016]
    6078:	f9400007 	ldr	x7, [x0]
    607c:	d53bd040 	mrs	x0, tpidr_el0
    6080:	91400000 	add	x0, x0, #0x0, lsl #12
    6084:	91004000 	add	x0, x0, #0x10
    6088:	f9400001 	ldr	x1, [x0]
    608c:	d53bd040 	mrs	x0, tpidr_el0
    6090:	91400000 	add	x0, x0, #0x0, lsl #12
    6094:	91004000 	add	x0, x0, #0x10
    6098:	b9400802 	ldr	w2, [x0, #8]
    609c:	d53bd040 	mrs	x0, tpidr_el0
    60a0:	91400000 	add	x0, x0, #0x0, lsl #12
    60a4:	91004000 	add	x0, x0, #0x10
    60a8:	f9400803 	ldr	x3, [x0, #16]
    60ac:	d53bd040 	mrs	x0, tpidr_el0
    60b0:	91400000 	add	x0, x0, #0x0, lsl #12
    60b4:	91004000 	add	x0, x0, #0x10
    60b8:	b9401804 	ldr	w4, [x0, #24]
    60bc:	d53bd040 	mrs	x0, tpidr_el0
    60c0:	91400000 	add	x0, x0, #0x0, lsl #12
    60c4:	91004000 	add	x0, x0, #0x10
    60c8:	f9401000 	ldr	x0, [x0, #32]
    60cc:	aa0003e6 	mov	x6, x0
    60d0:	2a0403e5 	mov	w5, w4
    60d4:	aa0303e4 	mov	x4, x3
    60d8:	2a0203e3 	mov	w3, w2
    60dc:	aa0103e2 	mov	x2, x1
    60e0:	90000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    60e4:	9135a001 	add	x1, x0, #0xd68
    60e8:	aa0703e0 	mov	x0, x7
    60ec:	97ffecc1 	bl	13f0 <fprintf@plt>
    60f0:	1400001d 	b	6164 <spt_ComplainError+0x1ec>
						g_last_error.file,
						g_last_error.line,
						g_last_error.reason
		);
	} else {
		fprintf(stderr, "[%s] error 0x%08x at %s:%u\n",
    60f4:	b0000080 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    60f8:	f947d800 	ldr	x0, [x0, #4016]
    60fc:	f9400006 	ldr	x6, [x0]
    6100:	d53bd040 	mrs	x0, tpidr_el0
    6104:	91400000 	add	x0, x0, #0x0, lsl #12
    6108:	91004000 	add	x0, x0, #0x10
    610c:	f9400001 	ldr	x1, [x0]
    6110:	d53bd040 	mrs	x0, tpidr_el0
    6114:	91400000 	add	x0, x0, #0x0, lsl #12
    6118:	91004000 	add	x0, x0, #0x10
    611c:	b9400802 	ldr	w2, [x0, #8]
    6120:	d53bd040 	mrs	x0, tpidr_el0
    6124:	91400000 	add	x0, x0, #0x0, lsl #12
    6128:	91004000 	add	x0, x0, #0x10
    612c:	f9400803 	ldr	x3, [x0, #16]
    6130:	d53bd040 	mrs	x0, tpidr_el0
    6134:	91400000 	add	x0, x0, #0x0, lsl #12
    6138:	91004000 	add	x0, x0, #0x10
    613c:	b9401800 	ldr	w0, [x0, #24]
    6140:	2a0003e5 	mov	w5, w0
    6144:	aa0303e4 	mov	x4, x3
    6148:	2a0203e3 	mov	w3, w2
    614c:	aa0103e2 	mov	x2, x1
    6150:	90000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    6154:	91362001 	add	x1, x0, #0xd88
    6158:	aa0603e0 	mov	x0, x6
    615c:	97ffeca5 	bl	13f0 <fprintf@plt>
						g_last_error.errcode,
						g_last_error.file,
						g_last_error.line
		);
	}
}
    6160:	d503201f 	nop
    6164:	d503201f 	nop
    6168:	a8c37bfd 	ldp	x29, x30, [sp], #48
    616c:	d65f03c0 	ret

0000000000006170 <sptBytesString>:
#include <stdlib.h>
#include "types.h"


char * sptBytesString(uint64_t const bytes)
{
    6170:	a9ba7bfd 	stp	x29, x30, [sp, #-96]!
    6174:	910003fd 	mov	x29, sp
    6178:	f9000fe0 	str	x0, [sp, #24]
	double size = (double)bytes;
    617c:	fd400fe0 	ldr	d0, [sp, #24]
    6180:	7e61d800 	ucvtf	d0, d0
    6184:	fd002fe0 	str	d0, [sp, #88]
	int suff = 0;
    6188:	b90057ff 	str	wzr, [sp, #84]
	const char *suffix[5] = {"B", "KB", "MB", "GB", "TB"};
    618c:	d0000080 	adrp	x0, 18000 <memcpy@GLIBC_2.17>
    6190:	910ae001 	add	x1, x0, #0x2b8
    6194:	9100a3e0 	add	x0, sp, #0x28
    6198:	a9400c22 	ldp	x2, x3, [x1]
    619c:	a9000c02 	stp	x2, x3, [x0]
    61a0:	a9410c22 	ldp	x2, x3, [x1, #16]
    61a4:	a9010c02 	stp	x2, x3, [x0, #16]
    61a8:	f9401021 	ldr	x1, [x1, #32]
    61ac:	f9001001 	str	x1, [x0, #32]
	while(size > 1024 && suff < 5) {
    61b0:	14000009 	b	61d4 <sptBytesString+0x64>
		size /= 1024.;
    61b4:	d2e81200 	mov	x0, #0x4090000000000000    	// #4652218415073722368
    61b8:	9e670001 	fmov	d1, x0
    61bc:	fd402fe0 	ldr	d0, [sp, #88]
    61c0:	1e611800 	fdiv	d0, d0, d1
    61c4:	fd002fe0 	str	d0, [sp, #88]
		++suff;
    61c8:	b94057e0 	ldr	w0, [sp, #84]
    61cc:	11000400 	add	w0, w0, #0x1
    61d0:	b90057e0 	str	w0, [sp, #84]
	while(size > 1024 && suff < 5) {
    61d4:	fd402fe0 	ldr	d0, [sp, #88]
    61d8:	d2e81200 	mov	x0, #0x4090000000000000    	// #4652218415073722368
    61dc:	9e670001 	fmov	d1, x0
    61e0:	1e612010 	fcmpe	d0, d1
    61e4:	5400004c 	b.gt	61ec <sptBytesString+0x7c>
    61e8:	14000004 	b	61f8 <sptBytesString+0x88>
    61ec:	b94057e0 	ldr	w0, [sp, #84]
    61f0:	7100101f 	cmp	w0, #0x4
    61f4:	54fffe0d 	b.le	61b4 <sptBytesString+0x44>
	}
	char * ret = NULL;
    61f8:	f90013ff 	str	xzr, [sp, #32]
	if(asprintf(&ret, "%0.2f%s", size, suffix[suff]) == -1) {
    61fc:	b98057e0 	ldrsw	x0, [sp, #84]
    6200:	d37df000 	lsl	x0, x0, #3
    6204:	9100a3e1 	add	x1, sp, #0x28
    6208:	f8606820 	ldr	x0, [x1, x0]
    620c:	910083e3 	add	x3, sp, #0x20
    6210:	aa0003e2 	mov	x2, x0
    6214:	fd402fe0 	ldr	d0, [sp, #88]
    6218:	90000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    621c:	91374001 	add	x1, x0, #0xdd0
    6220:	aa0303e0 	mov	x0, x3
    6224:	97ffebe7 	bl	11c0 <asprintf@plt>
    6228:	3100041f 	cmn	w0, #0x1
    622c:	54000141 	b.ne	6254 <sptBytesString+0xe4>  // b.any
		fprintf(stderr, "SPT: asprintf failed with%zu bytes.\n", bytes);
    6230:	b0000080 	adrp	x0, 17000 <__FRAME_END__+0xf8b8>
    6234:	f947d800 	ldr	x0, [x0, #4016]
    6238:	f9400003 	ldr	x3, [x0]
    623c:	f9400fe2 	ldr	x2, [sp, #24]
    6240:	90000000 	adrp	x0, 6000 <spt_ComplainError+0x88>
    6244:	91376001 	add	x1, x0, #0xdd8
    6248:	aa0303e0 	mov	x0, x3
    624c:	97ffec69 	bl	13f0 <fprintf@plt>
		ret = NULL;
    6250:	f90013ff 	str	xzr, [sp, #32]
	}
	return ret;
    6254:	f94013e0 	ldr	x0, [sp, #32]
}
    6258:	a8c67bfd 	ldp	x29, x30, [sp], #96
    625c:	d65f03c0 	ret

0000000000006260 <sptRandomValue>:


sptValue sptRandomValue(void)
{
    6260:	a9be7bfd 	stp	x29, x30, [sp, #-32]!
    6264:	910003fd 	mov	x29, sp
	sptValue v =  3.0 * ((sptValue) rand() / (sptValue) RAND_MAX);
    6268:	97ffec0a 	bl	1290 <rand@plt>
    626c:	1e220001 	scvtf	s1, w0
    6270:	0f0265e0 	movi	v0.2s, #0x4f, lsl #24
    6274:	1e201820 	fdiv	s0, s1, s0
    6278:	1e22c001 	fcvt	d1, s0
    627c:	1e611000 	fmov	d0, #3.000000000000000000e+00
    6280:	1e600820 	fmul	d0, d1, d0
    6284:	1e624000 	fcvt	s0, d0
    6288:	bd001fe0 	str	s0, [sp, #28]
	if(rand() % 2 == 0) {
    628c:	97ffec01 	bl	1290 <rand@plt>
    6290:	12000000 	and	w0, w0, #0x1
    6294:	7100001f 	cmp	w0, #0x0
    6298:	54000081 	b.ne	62a8 <sptRandomValue+0x48>  // b.any
		v *= -1;
    629c:	bd401fe0 	ldr	s0, [sp, #28]
    62a0:	1e214000 	fneg	s0, s0
    62a4:	bd001fe0 	str	s0, [sp, #28]
	}
	return v;
    62a8:	bd401fe0 	ldr	s0, [sp, #28]
}
    62ac:	a8c27bfd 	ldp	x29, x30, [sp], #32
    62b0:	d65f03c0 	ret
    62b4:	d503201f 	nop
    62b8:	d503201f 	nop
    62bc:	d503201f 	nop

00000000000062c0 <__aarch64_cas4_acq_rel>:
    62c0:	d503245f 	bti	c
    62c4:	d0000090 	adrp	x16, 18000 <memcpy@GLIBC_2.17>
    62c8:	394b8610 	ldrb	w16, [x16, #737]
    62cc:	34000070 	cbz	w16, 62d8 <__aarch64_cas4_acq_rel+0x18>
    62d0:	88e0fc41 	casal	w0, w1, [x2]
    62d4:	d65f03c0 	ret
    62d8:	2a0003f0 	mov	w16, w0
    62dc:	885ffc40 	ldaxr	w0, [x2]
    62e0:	6b10001f 	cmp	w0, w16
    62e4:	54000061 	b.ne	62f0 <__aarch64_cas4_acq_rel+0x30>  // b.any
    62e8:	8811fc41 	stlxr	w17, w1, [x2]
    62ec:	35ffff91 	cbnz	w17, 62dc <__aarch64_cas4_acq_rel+0x1c>
    62f0:	d65f03c0 	ret
    62f4:	d503201f 	nop
    62f8:	d503201f 	nop
    62fc:	d503201f 	nop

0000000000006300 <__libc_csu_init>:
    6300:	a9bc7bfd 	stp	x29, x30, [sp, #-64]!
    6304:	910003fd 	mov	x29, sp
    6308:	a90153f3 	stp	x19, x20, [sp, #16]
    630c:	b0000094 	adrp	x20, 17000 <__FRAME_END__+0xf8b8>
    6310:	9135e294 	add	x20, x20, #0xd78
    6314:	a9025bf5 	stp	x21, x22, [sp, #32]
    6318:	b0000095 	adrp	x21, 17000 <__FRAME_END__+0xf8b8>
    631c:	9135a2b5 	add	x21, x21, #0xd68
    6320:	cb150294 	sub	x20, x20, x21
    6324:	2a0003f6 	mov	w22, w0
    6328:	a90363f7 	stp	x23, x24, [sp, #48]
    632c:	aa0103f7 	mov	x23, x1
    6330:	aa0203f8 	mov	x24, x2
    6334:	9343fe94 	asr	x20, x20, #3
    6338:	97ffeb80 	bl	1138 <_init>
    633c:	b4000174 	cbz	x20, 6368 <__libc_csu_init+0x68>
    6340:	d2800013 	mov	x19, #0x0                   	// #0
    6344:	d503201f 	nop
    6348:	f8737aa3 	ldr	x3, [x21, x19, lsl #3]
    634c:	aa1803e2 	mov	x2, x24
    6350:	91000673 	add	x19, x19, #0x1
    6354:	aa1703e1 	mov	x1, x23
    6358:	2a1603e0 	mov	w0, w22
    635c:	d63f0060 	blr	x3
    6360:	eb13029f 	cmp	x20, x19
    6364:	54ffff21 	b.ne	6348 <__libc_csu_init+0x48>  // b.any
    6368:	a94153f3 	ldp	x19, x20, [sp, #16]
    636c:	a9425bf5 	ldp	x21, x22, [sp, #32]
    6370:	a94363f7 	ldp	x23, x24, [sp, #48]
    6374:	a8c47bfd 	ldp	x29, x30, [sp], #64
    6378:	d65f03c0 	ret
    637c:	d503201f 	nop

0000000000006380 <__libc_csu_fini>:
    6380:	d65f03c0 	ret

Disassembly of section .fini:

0000000000006384 <_fini>:
    6384:	a9bf7bfd 	stp	x29, x30, [sp, #-16]!
    6388:	910003fd 	mov	x29, sp
    638c:	a8c17bfd 	ldp	x29, x30, [sp], #16
    6390:	d65f03c0 	ret
