# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Procedural ONNX graph and weight builder for Qwen3.8-Flash-Next RC5 (`qwen38_flash_graph`) + DFlash2.

Constructs `embedding.onnx`, `dflash2.onnx`, `model.onnx` (plus context-tier variants up to 256K),
and all external `.data` / `ple/` weight files directly from the two Hugging Face checkpoints
(`Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-DFlash2`) with zero dependency on any pre-exported
`.onnx` model.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

ALIGN = 4096
PLE_INT8_SCALE = 0.000703105776
STRP_MAGIC = 0x50525453

# Calibrated per-channel INT8 KV cache scales for the 12 full-attention layers (layers 3, 7, ..., 47;
# 24 vectors of 512 float32 values: node_GuardedDSA_{4..15}_{k,v}_scale).
_DEFAULT_CALIBRATED_KV_SCALES_B85 = (
    "c-q8$_n*||_5IM6vh-zG$^y&M8G46(9$;ZW#5Oh%8SE~IQB>B75j#3oELf9WHO9n(v5^&{G3pv4mLVE7jx7>(ED_hPSibk1d*J)|"
    "5B%Z%de6M)U7q*T=ALuUz0c;V=CG_=p}Oj=R<V3i%eZ0C_AveaPI3B4t>U%s{v4K_*eTWxX%$DW`#G%W+aXrHwIhsK@!xRkUX$Xo"
    "`#ud@zkWANzp6`Iw*FtC_))i*IILaV`gMqj*}dY{MZdCh2E=72j*J^$OUDi)hQwvfedESnRdMw4o-t9`K5qVnpPxu{kJ-#Op{}W`"
    "eJ-`FbF6FnBqS28V=?!A$Yx%$-zx2ROFQ1e_Mh2zob#paJJ>lR>{u(?+Su05w%#$H#OMFo?<4K^A@;eQo!{QJF1B^&_eA1<_KXg;"
    "FX6otiKPA3%C<IkO;_GCoB3Zz7q(d$KDBdu*s+ebeQ)>eW#@EcdFz_C+wZtv+n6r=V4pkKvA*^@-VNpL8dJ4D1l-%7+f!xhY~>uS"
    "XSdeYQf2IE_xhim(~|Fl?{!VM$9GnaH|-hk*?yaComhU<S6x#H%aTa^Y}*I6mDzV|heX2mU)k^FddIH=o>9#8x9{U@d(H0W@03X3"
    "dYsda=M{55+4l-N{)O#74*BF8!OQzic%b43`@N^$5p{+7=@Pw8@r+LP4zII~P=-X}ebx=$9d%RJ)XmDU!;YcOI@|RhT3z>K`+M0E"
    "RzIkN{(J`dLO$8n&O?3tkNqQEc%OYERr^I~Z1|_$_jCLHk!{$29B>|fLmBFtO6?wHw!LTPb!NR~GpO&V-@dcsZKJnyAGUQdH`vPX"
    "jXkTqos0XR?@hJyQ6JrS4(h%>HN^I9)en2xxp;O9wGHaM%(hR|9|x$;+OV!RocEtl*QP9PyncLKQ9U58xT$k2{isiDy1R4y;^nG1"
    "eLow2@3pk!gJS8zHul>98$YdM)35BBUN&A1OT=aSTHk5-G;D1AE-V}WY)CvliSLw{{dy=rpl?h(KG3$xxctWcvFVeJab88wSoPav"
    "EPuJvFaG>u?o0L87j1jh+V}(O58XMoiaCtCUa_&EXm!_-ZHIP6zpQIQzs?7Y<6^GV>bz8AD`lUp%$yqvcCXKQj*sns2aKo2hW`YN"
    "*L9n=+wpg-?=;!IVEykKjvvewiA0fO>c%^U+I86a@K_@8Bl|4+L^1caorCd$^1Nx~z}UmQ^iBBso98rseGc$>?G^jI+@AlH?T5ws"
    "6vohV_W2z<*T=t?p#|rIZ6!GVZonMr^HmGYKOO8IAG4gr+?!#`dc5Bcx=%ZM|8MOx?)@XLscUNHdzCd~4naSDkMDu;=XF!U`{t7^"
    "ScYQm<A5^zeDY7$Q6lk^)lYN4nDO~+SoE>hk#iqDFMjSz`@M?&I-BVe%bLFrKCkVtXEkfS-EQOMLw-)xzGijbRds}N_P6Uk2^j0E"
    "YViFn&T*)lx+aXF=j}V{@CDV^cWlQ8Do__+2A?;+wLRvyZ5)H>U-{&BR(AA}f*t!_ec=bzQC-u&Y>%?xUSF}zvKh={L#>WK&^(KF"
    "eL0{$^2rXGk1%)qM{Sbi{DwI_k@(1tVgA{ncQ3K~V4iNTIp#CY9q2EpgM4zl?SE(I|Jt@e_W2k)XM*k5+xA;Kc8+~N&OV=O?RT<$"
    "Pg$KEY2R1cXIwkOHvB%>zAv%;Ks&#qm9vk1Kh5^*?AWhtn};%0*EQ`IcC0};2H3X0ZTs18XW4IA`;0P-3g=GP)7o*39iL>^qrd#Y"
    "e#_eaNc%p=j-P0spRxVZw%^C<_9ELyy~7;a_O$cvvi+G>X54?dZ42!lm)rgT{+>wer)STx{f)NG(L41IPj^b${%HIc@tumf^X>Bn"
    "`@EZduC(no``pcr9m@AA=1#Hwg|;1K$B(ydAm2HkoM4|(w(ISj3vJ7V)<<IwpKrha-u6|tM?Lfmw;p$n>b%a5A7|T8JHElbqmFO1"
    "d+lcX2HV!yZ<pBj>9*h7_S39A&agT~dsf@A8*RU@)!&rR`8bq)v>m$!{nYla>D@=$cDrq-*!O9+EwgQ~ec#)*ee83!?Wfv3UbV5k"
    "!an1iTUZap+<5zZjQy|mv)}i%?^m#lbxr%TZWD>&_IZdM-`zf+X8SRAUQhds?-$sH`<-BIcDjwj3HJMac0T$?cRTi5`@V;LUS``<"
    "tb;`29NVX12iSd<+3)l0oiF0MZ!0;>j!n1U4rITnYdXX}|J(LA+jar_g3Wif$1}RBy|1!ufA-^SW-rXgwq0!7TE25#(;|)mv!QIi"
    "$o41Mb-%LhAnQxzc5FJwLS55x+oss(%dOlc_PN&X*Tv3TWyemk&zIP-Pwkqa81HuMCe~9icctz3Wcl*RwCyMIdG`NdpHYT4Y>XUk"
    "pRZthCKC8>7-!dAW_5mowe=nvrweQwWn0F!Ywg(Cw!ee@ysioT{7lt3_J6YL&$HtP*>ldf`=C8$Te<6OKThw2{#|OnA7Oj6;~#C0"
    "wwR^gZ_@KHo+sLGfn_M>D%jV(jwaeY_p*H(`+c$9Yj3sRQoe5@vBdT>*;Y87NE~F_JvP@a;8;SPW4=iw0>?ojfq4dV(IIvmV<~6%"
    "#_{u6m)XoT)_)>_ak;|2<2k>x?>XC+vfSCs(Y9euc+fWV_l(|orfpgE$ul?))HThra~Fn{kDkJEV7%8g{Yk%PIY#oyakih#KAcaY"
    "yr|=cU?*^2*EElF(T+9IzDJH-pF>Y#nQ(1g)BXDGe%3=VH<^6~<JSA&OglD3b+JF|yO_hdS92^hHmtYrk7~@}Jr1+ahpHbO%kg8l"
    "L;ZX|+s?H%pJUr*JN~}r_nx-j$o^i}wBEK8R2O*WrB)9UIM4X^HFoYo&F>>QcJPkb%oXf^=(l+9L}Cp4H^zeZk=geBK>Pd%?T&o%"
    "81+}w?Rk7Q#z{7VG9RINeT8i`tY7qBpRbl`P8iO)9M48S_i=lN#>w95FYDR2G56!Y2xY>YmPiZ^4;^xj+WbKEp{dq}i!=|Te_v$f"
    "9cJGbs_sVEhQ3*2{rU)9bB6k0U|-5+_P2UP-B00r<&&6`(H?v0zv)!l*V#2Swm+Zq3C0xOp|RmLJARFA8TI)^e2;AAQX3Q1noGTZ"
    "`@CMK^3T@0p04(uqq%CHUAu+lNEi08&zOtGvVRnFm+<)*Z<wc3wX3Y0`S9?-Cql8q`f$&wcZ4?%dnjz(aDUkQzfXtNzk4Yp8Xhv6"
    "`sJ`<&5NOU=hNY-Jq!!a`!tMM@RzXhpr^v}8}AHTdzwvj_M>6#_7}r{*F6yOlOGCAm$r_XTEozfPmUXoG<*5kxn|!d<K`ne#Lutp"
    "70a(FF`Ipq+2`ZpFa7+wrm?meK1lSBu%*)l*pG%840qNw^^T6SfI&x_{RCSZm;v~^+&-slA8hMKjFrtywCiCv!VW`!ue9I#+V?c="
    "_XjE}WUIrT9z?8QbD^C#OxKs$XV`7v8+wrofPI_I;91`YAC9u`18C2|egjwFcx9l<3qE5kuXiliP37rqcHkiU+%G!57#4S|=_Sn4"
    "+sZxO_9OUxRSj?|@Iu1&9fUvH8(wjYUe-KHb=IAH0m|f<c1#2YscRapdc%Hzoiow)Rq^?k+NgX3Xyc{}W2{W5pGxu-!1MVe?Cege"
    "n?b^`ZD_-Lxd-rhC_ijuo1^VMo%Q^Q;z(eJSK7UMh=1uSdwHPkI|&c<6W7rrIyaIc#s^1|&5Vf7|CEb67|!~0Zel3w85|4hGMm|5"
    "_v+8~u(`zcW4T9v0Ve@G;BC=|yaO;&Su-$vKk`A@%rHBrjh^4baMyR@3~;X@tXH?ydyD6Y#QOPUFSR+&o1%B=rtj!02io`1D(^t{"
    "LDUu60Q?c^X5Jg0hqGSz)W%Ww_(lWnHzYp4|M0kVN^is0<KwdRU1E8UDe<CL$Hn4D>9}I)z_|I{>e$q=BHnsgkC^#=Mch2+!?3yK"
    "$oRs`-DCR&6XV(!tKvRZma4Nx#d*Qrso_$?@b=7={p?zN9~f7xX~91KOF!T1s|&HF*~(U@?-Fz6lfZLd6JMhb6m##`_u+Ql^ThtR"
    "rm+E-t)p`d#O3Ip|0MRpxJSP!=DLwT%O|@FufHiyYrD<=lf@0Ti#u@~Y{t5#4+7?aeDWpYU7su74%j;vhdCbH-v{;_aCvRxt;e;{"
    "xIsUkLhd0|`x&_?;OTr4xD?mz5RdnDaQ^8BVaNgE&%UtZZ`%JyYa8$26v2UD41P!3!MV299Fy5hNpw8fUmQ~lJNCXk>u2$9y`tmi"
    "c6Qz{&Tomtd#W>x1#leSiu2paSD`MyB8P$J41Kq~;q>o`rJYB_{E0d11KV%4Z5z3PVy>0A)b`^4%E;fO3!TWPIiHC8Op4C`{YUdj"
    "8`b|fd*9D!v+u;?*$l?Wc5*7t8;;<d<G6i?IEoL+Z5DHH*t|G^dyHZ51KCVda31w%;rNbh<HiQm@xO@cog-;${o_+|mUwS)L71n&"
    "1^Adw+V9)#USEZsTtr>d8&=OnJ`0?{x=kMkAM+j9R&`ASRQBGQ-%G_C;u_S$5b=j6(nc-jj<@|>;ta>@R|`v=N^D!#bd>!DT)i?3"
    "eHXY7w)Z8r-`9RyLmSEM3*fs|#3OZ0(`k#>HJzbyZnE<i*#_HjAL4*)<{8?pWzDb||6#}Olzok7UL6)cw-4<XV5_ob;P|77nSq;d"
    "za498>^ESKi)s6%3kMtSJzSXa@-S=hX?l;tX>)^ns%tueHg7)peYKZ)PrE)4hb0oH+B0EmR@(0~?3qXD{jS_qAHZhyslVCx-v}=P"
    "8{fkBShopS^bWgrv5oJM_I)*NOn(R1*Vo&3)bRtvtFRLjiOcNR0xSC^;y{j~U02Lq&->Ihoo@Tz+jgkz?Aye-9IZC^BXJq*aquX("
    "AIc2OQOx~S@3(|)T-S7#@a>suvx9_pPG_CKrc5Nzb~9AoYdOvwj|5_gV(u<-3fT<!%*C`#vl+B;TAae!!t@WyW{twkhtZZz7uJh^"
    "SjRr&_q)R0Zx#Ci`c^hm%YNe+z8~>6%9YI=XLS#IcAafE@%<ACv^V&d>u3k(lNXRH05+&=y460PKn@9IbIdu{zBk&o(#o>R?)L}2"
    "5A1dH*L9n2vE#rH_wfDFh2g9(;7@Q;ZXcqK@3POg+V-eD<05<R;nwEBTkFIJ-NE}dHXNfmy@Yj&HcuC(Tlo)XA1LOo<`_iZMLQ=F"
    "z(yB~*VxzQaquXI*l)iRkA*&aS9I()$?m^WoYC2A^SY)6;nv^U^9~6+%UaCctUi1Zc_Cnkx~8jy$M)J)#=550S(lDkFA{h67mZ_#"
    "?UyuGPatoQ&74GD6mv;FiT(@j;5Pn-`{a{<<oo%YvO$>e8ey*6h&$J9+S`ub!Ty!aT+Zv!rp4SOaW-32*Le4TvA@(cU1!^i9OtO#"
    "E$eIa4)}gMIU=-kKKU@mXd<!NwwpP&d|q3nx|k_!y-v9Qa-Dy@eSU!4P+ild9DhC@&SzPjUpbO<YFV>$5x`u<+<MMk_$^)d)h_NV"
    "k@&T+@n7w{!-eV4o@bCZbZ+G`)eXw9G=evD9u?U23)Wd()4_K9Z?@m4F}sfM<yie(@wrdg{t9B`x~98@yU!1~Z_nj9`Q#Pij8+Ku"
    "|4!qlKpsDl_yg-BpWKJ{@V0rvKA&vcGxq&#jU|k)m6|th)4bZB{kN`ZGy9;oNdw1xK6wM@*kbMgwf`+D@9pA0&>tFE_s-kq$T2$i"
    "2yXfuVg75a-(O68i@sdUtt5w1*Yup)aixu$3&|@La|hd2Z{@t+t~pA4&ZYX#xXH?NI=_2+*J}=28c@eLCtWy$ydM6~(C2(i{H#8G"
    "rP|<p;rsJ6uN}{N%Vw~@lXH&E8|+WT+;ys#N9@_RX|6j_?Qo?14ld&@m1ncfPdC`H2dx}u@&6HiD$IL6KK`b;^)JDFiVb$Z0e0@K"
    "d?)7-Phs1lPx##QCyk?Zq5U(rhSCmZgH)OgeES=rbm6O^xUf8K{I*40w)Lsdq5aUfdhokp-d~!-lXw0vJh`$}tUJ0#{Pl5fhVlo$"
    "3bhY^7d9MO3>#l-8OOAHJrw@>RmdOMEl$6`OZ@3K{o|*ZMTW&!#d}W8#q5>S<LdvKUH0sRxZ<XfF`Mref9dDDJq@hhQ$AGStWIX5"
    "!S@gBHr#Bder8`)65|zfgQDA%{e=l9N7o1Rkw1P+gpW3nm?m2m8~oS<=_7R8tyDhl4$<w*p%FerU_?LGgV)1$#`D}hg}vFuY+=~H"
    "qop<Ise272js(t67y6Qqu>Z2?*ku55wd4NDR)(&`uE3G`WFO)a{O-6oDZ9Lr{l1gy=qdl%p!m=st#zz(bdDlLY+Bbef|$3MYZ2>H"
    "RpJH46NdsTxxccR8jDozc;bSxX7Cw9S@-_VmBKmw<e&U1_<O?!?rO*SvwiBCD##H4OW?EP@EWzjH2Jr?688d+*<57bXUm@t-`G%T"
    "9bjiq6E3b0j}aqmez((0S$~ZU?d3b|82|lKe{m6&cCXLi*N`85p!gtQ6kwk*_WMu7An8IMU0<fU+aZ1g*r~VG7dQ&w&S~Uw+-KNB"
    "I+4!8`-9o99NVDIdPUcOwB>sgb7Q0X^jeU+^7cte+c8PoSaDjt#Q*fOXO_hB1LlWYFMBQ&?@Yvv2R#(>ldarsehjal^=??Pv{l@2"
    "-a8?i{~=s*`nMt1>&tM_b@seFzYA55b%`4r+r-UBY!4ge8ZLaiCF|?jIp2k*QC0EQ=i9}tryOY5{eYNwe7bE{5@TI*L%X=Ss(swt"
    "azNbrb)R4S`6C9mW`A_d+Q;_oq{HdNHm_?MObk}cRSQ>jp?^PJXhROh`Ik?mZ^8K69^8lBM)&BZ@%ICL-ahWX3EN8G`|qoBDl}$V"
    "sf}A$ziXkkhR+*i>!Ch2Ogs?$^m8-^bYXh}1B07{KOH__`@f3YO1AQu@Ihn#LV4VW++Tg6zqBZ=^uDhLaAl|~e?~9jY3G}O`G1rq"
    "06GZtL1=6SaUNV%11=Kp_xIqO9(cEeeCs>87Cdu+_4`kS{o$*|SRc%J6~}yxe{BENui91}9q)J3_{Y2j+%b}zEc!I&)e(bB#2;3u"
    "e<w624;2^og*b^)@oLViqzm9Ao-kVhWAS_57juBuDW3OEaNqxq;PY{7dj{sTPlW?ou<py6M{vwx&cXGw7IzeHTuCn2=Xcl)@9{q1"
    "W$RN(%_W^!CyfoAHJ`jK&SePe4Ll9zY1esm*4*%pG*0a_4@^=UPUjrp8nwav|0?Dx#OF?l&~EKmgSoz$+<@bSTj?jtCr8tt?|$b4"
    "=?{ai*Rc$24A^dYJ8w1pwAsv4@~w=wW5?0{T2-^w+AL?k9VY+FPP;a1b-dEfJ6E`8I<Yf21-GH^rM(LK5ZDubJYab1N3zF`mjCZq"
    "`Gdw1qr*p>%{*ksPqcltuuM)k=y>6^>xrWrOS)a2%^XEc<=FHTD;v(gj6TC+Znph?rfk?A@`ujSJ$JMBs}V*6Mw%rb-u|+muMrjn"
    "&O4Ac*p~H2@Oh}KRBZ)q?qY5;Z8NuL=gL;Qls03!a5=HG`}Cr81*qpe?EG=^K^-QaEwKDl*}uo@xxi9K&_4INUTV)Ds5)D~@@^|R"
    "guYJb9@cHTOYeTHz1u|k<Gc-E|ASwcFMIKHJ}aBak}m;Ifigjt(b({;^bz~Whj*at-Bab8Yp{FGldtYz`tgAain-n7=j=GMvEg$1"
    "--@{f)<^dgPjfo`i_jXl-CiSK{RP6RC);l`<%d0${lN7L&_X;W-{`?%*5cXZ3*fV>PaR}s*+8GZ`#8@RCS7Rvy2gG3&KV>P!UE!u"
    "Y-WKtq$|k1xZki=7-%H>gX>2Y5O36{j*!p$Jn~Y`onS2OMI2hp!H@Z{op-1(C-eh1%cpw|>$$9XsWAOn@;|3J7JZx!=efn)GWmF~"
    "wKhH3e!I!)?mX73`?U9Exm`Q4igkj0HZy~`KAYLk`r9+a!DuJ&5oOI!Xv_e!^|$-IDBf$Wv_My|AEI8tWdSo4a~}z7!tV<Vd?qzE"
    ";HaPraGX7dnh@Y8ubW$Wk7Djp-h0b>@Lgw;yKwxzPI&A-zMs!O=MxX34B5;(9P`=C%ha5>Uv(9A0M2c#A(jW1lP-*u5A;gujgGKq"
    ")!4gU$$NhN&97Ki$E-`#w+?4N!T3X;Z)_M$ZVz}4{SJK%K6IN~dCz?ENcs$b3F}k)%TJvp_vC%$0^-ed;X3(*(H_6q#cM$0lTRKX"
    "fBM7rygKrMz@WgDsoFJcYv*gP6wh`8{pRUHMw*LX%Rjx8Wp+FYy})W=>bvYe;9<?9*VsMN<c~16F=wO;D>Pr{I6ktO*=nnL9*2%("
    "<)gu_yOlZx=NlH#hn+60WdCqI!G%`0`&!>QRr-hw%j$FE2h>oNHUExtl5;u7%dfJEzVxB*!Uuo1l?`*rLmH!xiNifqSa%gU0q7dh"
    "KYUKOoo!m5x`cWU*JzzfJ^)-7`oOB1Ukm$RNKT@dYf}GTuku~S4P1DA>UV07XYBLs!u-Eg-5(~+*;;G2Q*_=UwZ+}+d(Z`g_i@b_"
    ">huOXb|veyvEfAOz{dRHY^$dpEPpn$oc+!5eOmm;TGn4NcMq?}|B?IlSMZ&(nMt-SA!k;fdW%{)^v7%l<yx!vKb`k2=8o2X;Q9Pd"
    "U2zla(&xj8PrehDjlVZ+KKG5#^qkp7m46Fk7Iciu4=#kn?B~K0A9jkBKYSH7-`XcOePTHM*;X-qQ&(zp$1G?WSGWEqT(ZRMtFtGW"
    "UEL#2KdEy(Yww})h((L-oONc~433kIuZ!(h>=n1ZG9)%WKGN|0$oNY?e^t#;!=(M_W8bl6l-Y5;Xluc)%O?j4E6%a&Ke2vQA-!;W"
    "VSM<CC(EY&fjVH|Z1{b%nLX&&O&7||7M!8`l*=zZRW@-q`MA5v{vTu4)a>%@yDqkkJsWoa_n32|`-{qiDLM$Bx05~Jg}yo9NB8A*"
    "=4ZzZt)*+|B5irOID{Tn##Z#FyN0>5w8o{hD|f7ckE)xU(_OfFc643wkoeQ1E$sJdmG^sM6z4tgPJ7xl16j6W4!(HsS!0OH^T`3i"
    "r){KF|C#zATxWI)xd_)#j}kU{J~($$9vw4uVVy62Zb)>^GkkmdD8>TVd4f2FKIAOGQ*0~gK>Pqb_LlV@D4wDf-ve<F!1b`<9cz5g"
    "X8^l9t{ZLT?=7wceWgk^cUR%>L9CZzu7q~H^AQ924#1T-*XBpn(|Eh@XVU%-B6k8mtJ$6GE2&yMAO7cF(v5T=50p=iWScm?>Jr_5"
    "=X^~zGg6!pG~gZ8?xpe8;}UjHJmW$;$KN@dnGk2Zv{3!1Uo8LffH?3spM=Ca<#F@7E#gJ5wusaJxjmFWI3{-Z<9}_O^@<yAYaQ2a"
    "ZxKs7^oc($>k*qe_KQt>cB9_ngj@d?o;udba(27e^jv8ypVThi^Ub(;&(w~w>aBrsb=vy+<0-rMHF4_{X!obajj#2JCmuN^KGeVR"
    "7k_?T)64YrqHh#)(}clWv#sh=z2v+8H28SL*!qP3XW7jD)>a>z?L1k&bNIy{uy*@AxDU4%F|qrkF>k=<i?KXPI-8GzYeObQk3H&5"
    "AGycv;91X*XLtW?l5r=_4|!~d*>Bb-;M@C#d~h-MZg9K`j&27rwCfSv9}do7Rn5!f0DQcITk0!~(fjg4W1M#s2Ufyym`^?xeBSC2"
    "AE>|>|2jC9cAf!oDbUvYoX|}?VmHnwC@1D!@XpXH`kd8+@8UlC_pJU92k>5imcZ<9-ZNGEWAOX5i(A%1YY$BOI&}uw%y#mK-tWH*"
    "j?2H4uHmZy{snpm=Wa(v_u+Nb*!(-S4)rOI*#ee#?&qgn_3U%RPEEo~tXt=Ze^ksxH_fFTqR;muIp3gs#T?FmHtdY~$|w7&ZQIy8"
    "jj+1t!kB{k6m%4mRZm@_$5E8hr|+7eZqZ|;;I~KJmP&7d|9~$UgMvN;twLkNKxzX}xAm!acy3v9-roOdv(ehIZJj^dEZs#y^Z$F&"
    "R2)id0?e0Bepd}GdR<dTYK|KlhS9#sCxN~CP~&g^CsxM?5Z~LoQzM*D4z*)brK?TaIb9K-fq73Dekm~kG~wyOH2I2-md<jCwB|$g"
    "xty4#tQk0N47H%poI_^}jd5L5cWPc81D!}eoBan=+Z;%}snt=nYgE6ec7HQ8+dUX>fcpa1xsUuf;rr3V0^mD}x#7}n!q!S!JB?zD"
    "!@5lqFm|cY_Gc~OyEHbGO3w^jK9cuxz5#fBxNP-#d_No4#8QsGrz&n>l)Wc3q*1!y<Lnr0zR#<nNwxnP>7SQKYnq|<yFPW2uy#s#"
    "7&bKCZ>}_?7t7Z_*!G9&9{UojLQe}Vwe@}Qood&kq(z@^_eMFpTHl!^-SuMPxP0<N@(MPt$-AIl!IQX#;ViZhbOx@yg`O5$tvHvX"
    "q)jaqHtEaztlM-j@8Ld3*UKUnA)86s^F|3@B%zazuH)^;cEdZNEzO1@<^#4vnTolDm3yFFKbab5=>35&@jf;eQe$lY=d3eeDd#2j"
    "Cq^sgzzK~KzE2BFj2Hj2h<<5&cKjYW4!lnuM897)gL>^LEiU@QP;vqJ<XrI>=a55in|&F}<r->enp?4r8XMZEUyK%Ro5?X%%nh{f"
    "y`@P&oJM=ZOxf}2VZQ}GRW~;5D?emg_RDMr92MT76S<ajVF25sKGmQ48yv^j%O{6PUtOmD--mq|b>Vva{psfg$6)myaSiGooE`Ka"
    "(8mDBdOt~1<Lu+=Nb*O;+*0bh%bG_j=4FUDfX<3bn8Eri=H^Ocu)@kcpW{4L8^tXh!oJ|~B$Mr3y2~eho31^E*g9QUqB&)>J#!wf"
    "1^46HfP*-;JTBmTjs?d-n2&y{23O_fI+Q$`kF&HrXQur=%l0EV2ApF8PqTBJjn+FKYkm20#h)CdKAK@Hk-ztm)V;V~XDRy~{NEn?"
    "F^B!$xyZD&(IWB+-d7h9>!ROyUp^$vTD+U$Y)+!~9Pwk$34-@HM!KJ!<1!t#l`Ljo{^?QZDh^Og$846__v1CL57(B_Se&L9sKpv<"
    "hqA5n$peXJ^T`UuPVFZ=4m>)aoUykL>ZqHv6-)I#bFB=E6o+vb#}49yFy~?J_P%tK=H&g@ubjs@hV|3f@GJ40yQyub5j&%gqfa~T"
    "&GI?Ui!LYUTFmuR|IUgVTyACDOK~`RYYd#EevL6%$Mesfut=QA?wSV=(KBXfE?uTRx>lUhO1;Zm>Lk41A(m}EdEzbWmuRj#j~cA{"
    ")b8SW_p{GKc`vk4s&+r|C5MY+I7;OlA>G~K>WibTe@1bwM{2&9LoI>-zEm94sqD9|4_T~v>?m@oYxntUShDv!W@i^e)gd2;%~c<S"
    "_QU(cjnhhEuGbG?h1rT#{d>pKTq16MqIEp_+wyqNtKH*<Ru!@5YWw_68#``T?(9n1PHV@)-fRl_13wC9je)PKK5pH#%<S<@yk*a^"
    "hU;g<re9UiZ#DXkn)pjUANJXfHM67ZMS=5D()tgv&wZj}({8e($H?E@LVms8^mDJ<G>uquXG}iqd|<wO^6TKfy_WLpcA#G{U4RW;"
    "Ev!*ad<_0!TM6)c|LFdqH^A#e*Mat=KfajzDmXVWIPQ%9h7Z^MT|<fYa^JR)t=n37c^LhCWzEPF(2eJ=s(~JOwD^d(gU8-}7ChF!"
    "tzrUNDc3<Cvt8jc94Z|G?B=%=6AT`2UWEU-nA?qb!+nE&h11)aP1-HGmc5U-hwiM)bfLF$DD-0;yFR#w^bo-C?}<-9oc$E~>KYrG"
    "8Pn>xrfqb8EwKMWX_AN7x!uTP<daEh0)W}RmaZRI8roX;A4l)f1welREaB}sE4sd-J^g2m4d8XQ*?-sqyEkIK$El8bEAPPs>Xq>v"
    "*L#d5#`k;(1Eeh}C%03~{fBYa#aws!GI#ok(Y{m38=+6R-?9(e*<;Jgq@ylj{0RDw<FFajsd!vUFUC}$z1@ch%=j5O27ksB`5;>>"
    "zXbFNBjSeh%42zt2gAl*_l3HnUkppH_$Z8SduN#V{Ij9(*T08(H$M_GCEE?3KN!mE+Qq;9;q6fVa_g9Rzhk`QhJS_4Eqlhy)7HP="
    "YaOc&X-i%Eit77A-H<Nixi-I>ivxe$-Ny4Y8>e$)X6XL0^rKGpc~Gnjd;j9kPb7jg6D90B9#0?DPg9C}sU}b4e$?T@YICiA=UUs&"
    "+NFQ-c!CPGVPD1Fq~wc6JGLNa;JTc&_{ElNyT*oY^2JwcKg!B8&d%u}9Y7zAzb@(rtvGl29EA9rD)I`@q%<~Ei4$zE92*m*U78l1"
    "=jdy5K$YElx9Ix$s$JR}U~K5zUFXu0dK%{iE92tlz7NnBImg#Zap<G;?gN;w!r%YM=$se)@z5HLRvb)cn`_5Yiw?gv;-$b}qzfaW"
    "$IOpmzenB^_qn$czX?1&pW`u`sZfrO3cCl!(kgpKd*;^gTqu~YC)u%OiV>R0vSl;l?VJ>GyvJvBVtJiY>7=%DJ&tphh;jPd?7BYE"
    "JXA-&2E5J$jj6G0d*^MwwsSG}Km!7fsD|U;%MK0MPL0keX%D(ap9_v)JOOx!Y-W~ypUe5cpM%(--Kbaaxdky8gURV-GvD$50rRKF"
    "LqV(5Ra%m+?5jS1c+3#~SMdLYzPGL9XU(m6=8=pkh7UZO`Mud87X^=%%7s-mchPompUA7iMDWSot9ZkUsJ*Xix{0=Xed=WS8FKQ8"
    "o=QKObLI=BzdcO;ztip7KihViwf`PwpPnOa>y64I@kjb=n^%vtx%UQQu5{rk#lRmQ9;m2eOlmfBuKeH+Dqi<Y#(%@^FKa$h*bDW3"
    "b6EN4X|mhuS+>T8YWnk_HCVUlLA%#&c28)H-=}>H%mJTEs`hZ=KY#z((yO0l=buG;8!?~Y>frBzUl(><ed<YJzSG1ZHOhaLv-6Lz"
    "=j}%Q0BlY73qdP-s%`fxcJ>tc4Nuf_4z}_hFYWP}^dCV3?|S_Cj2XfE!)NAs5>`t?e6M&F_?#c77T#mJ(^iHf$)RL3=Sh!$6@9H9"
    "qka-&ZjoETvD!M~Cf5b5P<+O<)PBS6h1MS!3mV&=6OqH>SB#lS7oN6b&_Dml+T(P_1FYM$QE?bo=)MO@H*x;1xN!Js(}nYuKVYq5"
    "2@aQi58d$N(hvMKcr4EnyYGB)P}c;<bhGJ8hR-imyN+B|V?(Wcme=e02I<xR6kO{w+Uo9XX^l^iF5^scERHoUl4kxq#+<`9+1PLZ"
    "`(&#2XyL<$=wr(#Zw))v+|S?857UK<iGvUaoi3atU*avqVQAAGYhIxSq?mg~IBWr9K(d)@<#PsK^;^E1^La1Q-<i$4qH%ew+Tljw"
    "<(n1n+iqrK!yoOua`lyJw!3Rj77*W;HRpI7{@z)O_oI#jSfV}!tUA_?uQ2?wl$hT68yp`Y%sfwei8GjE<Ns@Rj!~P<XP@7xop%k$"
    "spMB2EB26Pzj{|L18@+Iqh?4$KUo~!g;u6C$5Fa)q_C7@)@;VP82GeZhj5VeIlr;}TKcZRnfX|*v+HNbuYEDcFmfNbkABaYGhV7="
    "-vj2$W`+rqUn<ND+@4~blyiQ^upOcC@pzNlsaJQdCS~_tAstJwzA}Rxlb7K<X-?)y=QH1aA0RI066L43PI2g%Bl;>AM-AJ}`G$GS"
    "BT&rU%{UUj=C_<%9m8Kv|1h*S#oS`W(cfXu86uy*#}~Pts9xHytTZVT?Yt$b`_m1d0h^wqSg9rdCuU><`6TDH(yXIY?H=Mlmq-tE"
    "w79;JiiO`t<9sUdyK`;3sUA*NK7f6AKj(IrGp5LSm5DqLxlizp|35!P#-2MUKu6@<>9xwg5ZG^4)tt!r8#;RQ36Hh8k#idE;d+<Z"
    "<jC;9;9S`>#W0=2cnfIvp*hJXCo68@Dz=;J=H{w>ZYIw5|A9q{QCJ|1j`}-S{K(bP%*-NZ>SOjudxz!Xy6<A#mj55TNuJp2dq48c"
    "$Qh8$j8l#n#FWgk=N(MGp*}T3c@9p~oC=NG5^)NHto+Ma@ALlhZCLuE*&3CJSoKqj*mUWzxcS`o!zI1{Yc_OQteW|K*!s#Vq4?3~"
    "p|nG39QbJ0xUrYn+_mo*Ht!o(|E|QgZt<zo?s4tCr7?41m-zEpQ_U8iA6K{T6gS;_u6@5VE*o!l(Ck%l`sbtK&xdr2J<AN6%^ned"
    ">E{FM!Z*8a)84{NLuHFr5MLvf-tC8u!k~SLo8fZ>R>bxBWM}aLU8J#}CtuR=UADPnzDjZn?qhDF{pjei!r%bL$Tu@Cx_yQ`e7$J@"
    "*QW-{&)CwQ_m%9+cC-zl=}#Bt=>CI=3-U=|l=;#9>YauA1~O-XzfT4IZEo|I@;Cq7PP*M*DpzO4`Ocwl@Z1T&>IX<y3;el<VVd^z"
    "bwj6+Pe$pAQ6Kx-_fheuM_W-_@9&kApRZeVe_k(^yFN9JWp@qhezN-q5t9Mm!XFNvB%cIc8%aMF{AtD91YwmQ;E%E2yU6YyC_j8}"
    "#%}-%A;tq758@>}XTxxn6ZZEswRt;fA;wA<fOeln|Gvk2v{xJUU`_$VM|<7%7JlijoCyP@@ffvB!|xi8VQLdRcN*~|e1X6lu>aw+"
    "+p%VNbRQ>TAtn<ip<g<O0$s`;cKlNt=g^wWkVc`GbQjZv4NHj~fo))qxvk%yWk}Wbl8@QmxANL>OUDi2$&a22r61i8(tRHarDk{E"
    "((!hi?{5u1#mB<WKRpoMsC+)WaKL>bGyl7=;k<i7hYz=hecF5y_IVFl@4tn1f8QR;d%PB==D!H{T!&cx5A6C$cK-pf_PYifuQRQ#"
    "4)9zGvy<f8MlYZLi$5PZ+kovnPem0uzOv@&#8f-hl!{APrn>DftY1YeiGH#8xzX|&*GQ8Qh!4^48XIcFrR}5G{NCzM6^aF^RBZZC"
    "@&%0zf#+>m|E)AeEvd0V8+#e%NdI3R9gp_m7zO_HoD<#Iml_+otG#h8c%H20s6iZSbxre#rM<6q<@~U%1buj^$~jy<?lQh_eX28g"
    "0G~hF@SJoZ#j)-j+HAegRLxQ4>gQ`!U+C*g#f5mT42-93CDX-Y`g{mo5aN7X%M$DvduYrgHDC4P`1Lt*fMR_Hk{iX?bUp=h;tV@B"
    "j&of$ld}6P;uypnZ1%5k_^-&dc+Qaa(#j80oJ}?7cJPYng6jy7U&A>d^rPt<kEz<Byr=z_>)xGN_H~;EM2{`njpI6>T&Xy+0h&7|"
    "XrAuP+$iXah{Xc$xNg%ZX(!t78Q?EGMrpBqU!Zw<3h#kF=yP#@_6^s4O&9MQ6bsUi^Rn|Z<1{Z06UT{pbeeo(_fs$G@u+)9|2kPd"
    "wxNc%*U(OJzr+mbc59gn%I&%H=!13N$y~FsR?FuwjrJ{UXxLP+o!s^t$Jk_hPx(n#OV<nidCmXvgYGMT`B3S3&mdlLfAd1&lv>&P"
    "qoe~}Ctp-xUM|M~Q|YhtxP?KqO`(7HIQmFm8Z<htMHnG%!dcQ4PgGv6-y2TqKpPU8UD&wppSn<anP6LOkS`y&YVa;UjmJ}5Cwm+|"
    "$Gzk)Um%_IamoR80x^U8UC)u9bD?}(2QXiq>)nSd7gtB&rmW)7kCG-}9^*jkQ_~c`kdu~uhVqTgCzeYW_7=_=uY760l4kZIYCs(C"
    "+(Ml={9vv%y@)Z*$ltK7WGel{`Q#w!uP4dwzD4l^bJ>4@Ens`PzF{(T0AA;ZF_)p2`9kYwC(*ABTR)p|UmEa+=N-iT$0`>A>U9L|"
    "f5$G16?Z#VvHXbp9xs2}OyS&P8At29$~<aG^2x!(^{y{JT7BwJVe<pUwd^k2`(WWZ;K8ZV(C;oA{2aCCbotz-u<l@sd!C4Eg&7x0"
    "Uococy&7>g<LDc8{m2yc?<2@fxJKZg^l{sNfw0m+%!T0CWp~9<cz(9Vh6!QaCd8|pEIrG4tY6&6H6M4ePU=(pkpDou0&<RJGdIyU"
    "3aoeQac4_A@dU9acnR=Ou5&(AZ47_u6Uqtw2j#H*mECs@a~XIXJv8YziWfLW@ftPEp_<LyX~%(=zfdl`WsLpzylywJ-qBY1WJYcB"
    "46g-FYiwv>%uTwmukzBIOU=Cd(I2tTPfEi8E^4(f^^??e=aV;+>%#TGSd9$}6xV%s*r((I&TEOpCd~ne5BZy7b?#FB$^8|hjlR2("
    "I51t<VB0Iw1I;Iv^Sqn8+c{?_=KEphU~u33G-?hzANLCT2Q<*mfuQdow)6?B$E$4nBXg`2b9;pmgD?C)vF0a`(}I57;}C9E9V}&="
    "=acU!Mj)qtdz1X(ZwLc!WIID+1>GXX3w-z(A68e<bHF?tJofKOawqQ7K9Kwr`c5{pO}c=KlrQ}pV(Vh=-r)A{E#l(d5~uo@VrMS1"
    "d2}9QCmiRW!@QKPd4H5z8ppyXt3LM&?uWTg_rV-n!FUDmYMzJmeR7*VzieP1_y4OScg+hcAN^c$8{kL&X61NU9B-}iTWq#+trF)`"
    "r+E2$r6F0TcUh@esiVV<cbrV$d_MWxux0(uybyJa>G5Y?POR@aix4MxlJaW6ep$|Vp>*L8YsVwiccAa@9grt_Tgg&sdC%1U;1=aH"
    "dR#Fm_)kB4SFRS%C!1Gp&B^4d(uKpBXTr<>Cuy4wQ0@ctk!)D8#_HqFheEb%A#}L-zEC@TQ>f~{B`n()_V{l@|AXH)dwX-}aaCg|"
    "fBBE$;lp1FiN~7^lRpwpe6mG6YwwT4Q>lN2bMD)2$6CjMZ?=l1E8E3OPA`d*UWoDevdiPL`<BGa)9d1x;rqw(NnH$!SH=9l%vNmZ"
    "5r66DyFEJ6Y^0vD$41kx1cqI=DP^|rMB0Il87t|B@;qGoOEYkw*={M>xgkOqT+HpK{ByANR>^NSiM~SDZSKq$(2jN^?7DTE2GUpO"
    "cJlz*itv%UO*c@!`a#kU%n@E0ZqESbI9<5~f<0q6^|g%+2g`>QXq)@<hp9|Mg^9;%-GEto$C)gH{h!LuJtTUZEwE(9&KoS>afN)Q"
    ")v_ZO*tQpA3LFEEWId$|(-c!&s(1x_&ro+;pBhA5l&YN~pW{?v+0pdnyMMon?DFZ1vw&R--legjAF*s%bFk}A6P|z%f1zw?k56#F"
    "`oLY7&2iS$xN|LnMB*6xy;8i-aN(13;xs=#NPe^4)PQ>~$Fah9<@BfeZ_}(yl{}7l(CkdL8)A2>l`CSPe4|s<#w~^GhO@36L-b@B"
    "|NYaLIBW4t`wq@zys$mm4s+@r>~oG$dI`fJR=Xd0A>avMTIe+#TQ0EQvV0%=-?3wh6tg~-ee{Pj->zOh<L_bH)9;0=EAI~3&z=rd"
    "GoK36uev`}J@!brr~J?1^NIfm8^3+h@bo{!&yT$rHoyBySkn3D@Za4&vvK)+*yrf?!-m`54I8ijDU?3jE>^uYE^h5v8o!);Y+SbQ"
    "*>OXw`EhNRd*brV^JDWrlX1*59pk(+`^B~Q4*tcTk9w_74QHRS`B*vY1~9Lu<HQll-?&hm+;IBefx{55?i@=S`Q!)74?ETR=0y4D"
    "e`3xF&vh|D=akZa3r)UrO*5m%_LOTL!hAA`^HQpIXFk$FI(Mi&bE&ZYA@+TUauiGwF0K%6A0j;2PU~6h#O^(n6SD{TJNTEKm*^io"
    "FGWxK?s08db4Oh}Mp~8<>QovVDx^^w!gT=}8}`ze7(zch>=*A}EyOoq%tDu58C|P0S^Qf+%@6&=r%va0#2WdtN;GbVDb~MIxh-2O"
    "4ym=(&v0Sg{+jQq7)Jw5Q9cQc{XBba(%Se-tx+;c@dw~Q&gPg$&Ii|TMa9~T)qGQ?oSl1!&mF@yg$4l}8@SCee}H~yHn||=VJ&M0"
    "H#9}@RT=sJF)!4p%}S*0XwNwk`hDkWM#<;jNBRHzi%XiUSc666U-3Ty-pBI}^x?Q}Y#6HfdZ1!U_F{eIlhy3k{yyNc7jX><zrK@|"
    "75l}svmJAF3Om;9#r##!3#SV|R+lyJE??F0iqm~vn(O_^5$sqqUHGIYaT;uO_wB(~+KzYyei8SLOg3Cn%UsHyN53uY)pQ|E-zYF&"
    "y0F0Pp{dFTH;fo<-KM4V-@#sTy9&C~LF5n|<F8dL?@VGO_^i`~Y*<yZoLD}eTp}zx#*QCNO)ugn^T|o#A}+LNELKc^U+TE?$wTDh"
    "o-JD*`QlHeh96}|jyd-Ub(9{pnsz^8kCA)db<k6&DTPgDW0qJiRohYV^t(x?e>wd&?(f-&aW7P^>oT^(;^z(`UW1mln5!Z#gZ+#="
    "3$8bxr+E1>(l(<_kgI$O?Y$W<9fepB#vtH5a^Ie<xQH5i?}>a~y09P1pQ;_oYtn_D<Os5v3yCf3Q+voaw~#m^T^J`HIPz>kqy8rG"
    "xMT8BiW%r8TY3uZ&wR3qaUSrUfxmzsH=8LFp1f)%Vr0{W4#8tk`YC5(ov`{|vR`L07n<V&*v*5fiEV6{$vSjR$4vU8i@6W%_iC1V"
    "TggD;{B!~M<=S1odgSbL9%rJ)z;(*oHIezw&Ydtz9L>&L_op(ieKvEj{A-sIZ>I}q(pTs8HjSJN;-PU4Fsl1&7Z9s8HU#OIU$M_)"
    "l^1bO`i{XFrD_+5-<Y9i&nBm~^3kcn++(fodJ{wDljZVJ-Y%Z%YWZ7du|Ii?`82i3RPj2a7$X5Zi1x?(uBtg#T5a_CM}+SmqUHd2"
    "dEKUcc<*!pZGSVzbz{R^>Jo~%2gFl&zqS7Z<x(GOpBHgFy9Rub_1o3bVw_97z4B3v|24{$cN#Ssi0AjQcpJ|tYkoi)miv`U^H}mv"
    "$j^X$q^oM4B3JZ4#hI*g#3s4Vcmvnmu>UD)7K*u3sYS^rFV~npfVe52ywvJ*rF_-5NV{;d)@W!TFFE6-MRv|q>OEG~+{AiK7f!Kp"
    "ce}mgByx3iO{<C1fcX$31TDAcTYSXM8zSHP@5y-pKVmKd&w_PmR@KxI$7VB&?fY5Ow;=8caXgI;$Sn!I!iDPBSnJ?)yJxL(j;!Sz"
    "fV?CatNG+f<P0z``+5}TbN)>iz=@=1x(+F8zvmQVRZZT}+w?4Z&r`*<9;tk-$IH+C8_t1sO-bf2^*F8zgy9cn|8|b&3h@OKn9rfH"
    ";a2h$WzEPLy4t?KuG|AJDA(+{iZS@Rat!Qa=VA`Jk7FR4saG7tM#Ux^qdd6F#cOu4|G0~F9QpJ%FlV2?<B7_1d5o_8jr#EoIv?w5"
    ";D2zL@-ALv^@^OG8#ymN-RXbKoq#wC^l{H)vQ}~b%UMs24Qt3J;XB^XW1Md1e^$EinBw}Ot++@zS$$0r@OPfuAN}s@YUdYDCGYHY"
    "GMe1Wjy2cXGmo-5S<CqURBa9C71u>wpqQ(<(t_dtWOvnPot5n}@$i8f9<&4g$31rCg0TGH_rkz_jp5;2o(daZ%Z24PUKf_#_pk8O"
    "#|<ITP!f0C^kg`3?%js5zYTNFcqWWl@pG7V;xpla&?Y{$<=^4%Der}~FTQKP{TSAE=^VG7GBLJmJ1~~sS!y=8*)9zi#_9jOG;W+W"
    "Gk($IvUty_i{rEtXT+XY1H&F1f9dBtCY)*gXAEsy*qEzoD$K5&uC@FI(q=?HY`2d~$q%Gz_hEczF}FYMtojt}t0;dt?6^ZCwDI=e"
    "Kr8_}5jGs~qK#|XpPn0Vuyo&uA;=0-4`7~l&;8x*|KtMfYu6%P0e-Scc1{m+640!=z29BFwrbk!soL$z8GsnkZqnaR;Ti|U+-&KL"
    "C)m9vD#!dtzN>3WC&+fKjGm{wxBTE|#s?~ZIrkz)ANuYB)(d<mJM)0KfAiP2-%WUUY+U(hS84`QwKK&lOp@=fx4xIlhq{pOP@n29"
    "jM725u7;~j1C;N-r`fN2$WM;E>W47y88`y=cv<r}`u~xC+x50l+TSkJ9-xh2JJ+WcFs9x$6~n2ea7}iSng;g|PPF%|i04iiuUv}*"
    "i3RL`P4NRWqpu-&2yq_xAI~W`#EwC)3;*W+<YrtC(p$d#(TwqT%#meoMCX(`OMgCH>vn<jjKXjzcU#^II0tzFoHv*toq5My{%pJp"
    "a7Z?DjgHUf^BWuXR^GUMW8Ll6X7BvRY@?UMyqiA{iOPFI?)t~WiZ$)ylOH`0w)Wf}HXgJkJY4fZxaYc$ZA`X^RX=r)C%oGn5@&a`"
    "e*J1#v7;=``Tk2Ahizi{0bhqp9_|-!DsL4#w4WERowJAaS9```x5sqfUh$HL4~kniq+(OY!|ho7<)7cw*6O1TwH%K5doj<Q=cwqd"
    "_DynZL3fS%Ox6AyG0?=S(3yMAiO+0ah9>os;B!Wrn9q5aes-;|=dfeVT#do^7-L?ankB8xQR?sArIGoDJ`(7?;a@j9&CZ#uyl$2B"
    "YeWC}<dzPx=fnhayL)^QV#5<w#yzDcf?nWX%5`5FJ#OS9ji+jHd!Gix4q;4qEJ?YQ3pr0GXl!*QZ<tU1sND9*k=jvO1kWd0);x&y"
    "(AY3S^TZVC&=EuP81Vt-EzAK}_XGR@`mtl>Z-R4fdlQ?lsu?6cv%R>}YvQ(&AI&bA5}p5VPkx|2HBcM^xVhFG$IiPUw_xBi)@^!|"
    "`Sx4`(~fymd_P5PQXwr&f6jm4+tP(E7`J8kS9wT#k+1PJAXCb<T0zY2+N$xKCq2Kw|AOlXT1ZdORyj%zz*>9u{c*$&+r6XKX2n<@"
    "LVZ)JwmG=2q>S8($6WPS{6R~$lh5IZyBZ%o|N0#1K-w|)q5ofWv^syD>x}wbvCQhKrD6#FPVGOqE6-&BEYpd8(i`t630u|&>)X(*"
    "PBfc;fHdq`#tb?>ND_;rYL8GXejrYT<~N%e%@}RRZ|&?puT{J5aSqRMzh9bb9s#ezzVvmLj#cdRH0IL6+5kh}y_I+s7y_8oaXz%b"
    "?U;7}@u#k(g-zMf@ZA9FEVr!xp&CBjvgR+V(}janzBF@Er)u}7J)WvvXz#Ws_4A&e>|$a@-~{{MAWmGjDM`I5%5U|gSn3|owlj7&"
    "U6?|?1hzA<mgf$bV10C)@(zxWc6M*!uO8BD?jdYAKv){u)GpLcq-qZ%X0ZQrd!L!K*`ak^RkJtOTy#Hrd%N}=av9(@pm~75*mVjS"
    "YGQ#65Z}D5WD@O5Xiz<#`3U(Y_qKbqvTYdo0FSd>tlU~Vs<W9^$_M#t#d%cIR&<{(a`E><f27^HZqw;j?xUn(ND4>yv2%OV$K?It"
    "7&~vSat5HTf3#=7_kN)?8~ad~jd}+z121D^LY&G#Ym?`qYmCbop9{?}FmAfgNA=fRb=W?9{mmfej87N(8n#@bSoq0|Imdc&xF7sw"
    "jy2Pv^Kqvr=6Embb17=)(}iGj)$vwuRaT}%Sp3`#(xd!b?b`MIxLzT+8P^C`E0$oQus{6%J!Eq)l{RIG@-ZGJ9r+yj?SYrualJM4"
    "G0#CTgV@Y<BW<njW{6h-zU#zesoF_wud?R0iUYraWyShp-bTw+#`#*~a3<TsF?k2#zGAM7T9QPfg#F3$NDfx+$VJj)15Z!2=fQ@U"
    "E`9VQdu~#?hlR|G4-H8^IiF)KU6^CrMDlYzC!Vd`Afq*oPUAX*9)~`G+98ZrAN$T-<&(Sb((FLb+}O}lx|cDE<(^B8HDb@9YfshA"
    "6Gybb&JWV(j2BOtk$-d#Ytshtjr%HZL$zIt`U}!)EFxzD%?9eHthq*dj-8y$Byk24gtG_Rxrg$deZ7g9(ss<!I4f2E*we;vT5TTL"
    "Upzl!PhsgA<w2dTSoDnbIgeif$Kqpqf##?Q)ChnZKn~Sn?h(e>d;I<3#CRBks5i6&+S76G9M#KA&S|cz+EZMV=d@V-+&u9j!&ram"
    "!aV7;W@vsmK=)Y6Je~REY|R<_NGnlg_nY>Abo~953uJe#4?9JC>saNDMGlaZV)RpC*5W$8XJbQ0<~zaogg(jR;#aU<JjP?8*46{3"
    "u~ch1_OS9q@n#*R<LO0i6`CcCJ;c?cU2tt#^L+cS$cUSqtNaeL7(<j#9?ZI4{M>Bun1@g^g;)uX)j+<R-C38;GcBU89lSQ`E>$~2"
    "c^^Fg#{(5<J2sP=A^hJpHmnxcJxs9^dsCNFpPH|CU7)yzJ(RCyu3fW`+;my<Xw~mtI>-5ETvMN#W9N<HS~A|B7O;+eZ2`od9LJoZ"
    "(4QgSDLC<4N8cGveDcbWc&AMq*zb0;zuyZ>Hog#+_bH1%MY9zv{~fNa+7W)pc8rO~+s0+%d&Vzb{w!>`t(V!*UxaJtv}R7G4~{8~"
    "FC0)Fmz~(c?CP6C{$E{UapjEIc=n&-+AgQWt(#Wb^>bt0dxyk%Kh&6=K0OYoo)>@V=OdTfwvt0-H%?<*u50ppON-iGKCuDRgd>i0"
    "-KO@+Z(kvQ+i0!zHH5LVo-1$0F0HuhRD11;3HJE(Dbck_)zSo1(6<b{2!9o9z;t1h@FwDe=PECKn_c;&9S3xkmNXsR-s?o{m#Q5m"
    "{63!NZ7cag>y8blFQ}|}h;*bMGPj<uAzv*m^JJ}a@A;`cciBYw4PgT}hn@QTx~2n_i+-Yfo2}&w^!(>GM=GEFcxm<9E53EQeC>T$"
    "C-}|dF1j*T7Bm3vyYIwzOVw5j>-Qv&QP$kv>Lf`HA)7gj{zT*e1P1c;`bG)QwUs~fW932bDI5^wTP#;>Zg1vKw{b14cNc0NyuI2h"
    "cA;Ex*4>F)+@Cp!>mH;FT@_y+73*ImpKpb<9(!?(IA0%r9M>2`{IbXFgQH3FxZ?=uCk`S<v~E);X(75w2T-EC51ptfc5U`x-D3pX"
    "2>iy5HI?+kB8P?NjF_fe={tSIt%&zF+}0y5+xk-Y?z_K*s<-Ykd_6R_f9A7LI<h$|8~;YA44;PSSM`Z6OnN!|`m$~|mi`r*Ms<sb"
    "$$8_Q*1X@+7rzPJKYKqcKX^cF`lMt0`6uWLN5yQ{ljG<6-x603o*%R047(n^EN*PvEk4|0sA2fAHdc@P#h>q--C)*Ds&*QADUbih"
    "_}QH~BarJepPZoF>PM@Mz@Z#QZG~gsiHZ-eq+YzNd5m)Q^;fL-ApYj#0=n?y#V7Wqh5&xxb(=;=e_mnV=d)~$4dbKh2gb`s-&=Z`"
    "Qsw9y$vhgf77t??eZHBf_@dT|Q5r|RkNq!Dn-1Q=<6!2|kDV?o<eHq$>Gk7r*B`Z&?f~4!aLo+^sDXEF!~pRSqotWhX^tAMH3oiG"
    "-i^`JQLKElOC0*{6pf#LihaSlq-o{#AEC8K21;+xjs4!&WL==~HeJ5@ej0lxDL2A2)(3bAZ@WdT>vW+;TA-<%3+q$PS@?T(6u;Qr"
    "$}pUBr{@tFPF=*xN1*|W;xu+w&X<84yS|RcXvXZA4Z-*dXc2sU6RZKTNOAe&`JaMZu&&7&F5b6PJjp)dPexLQg}kZtslLqNfcDHM"
    "SJ<=mR=am(|3q8*d~g&wLhvSNH|K+^rD+%(eZ9h}xT@wW<pCKf4Mjrh$bwHAsF>J&nL8SOPv?iuSNt&6S>KCs@U9!Z&g_i48H>9!"
    "e?HD_Y}n0gr`_eJyF$3`K>C%knSThoEF?bl>kg7HY^id$ok0H?{57sY#hUl<dyM5;#oz>xAAQG~r?tL(q;A~vvcX?_uVMr;^r?Al"
    "F4l{?Ou1?<4b7{UDF*dv`CO}+KM?WVjSa{>zK&XfvgRkvHrvhivx&n}wZ}<+a13#!V~_ddK;Ub^`f#b*M#YOCN1F@r?%PUs=0Lw#"
    ">;3(n*t@aeX5!*4>(@x5d@g;+Zr5HCHeQdI;699LZEUznnB!R5q{!!0%$-J!EBu|v@9)0oE!4)lKWukm9>jQJT|VRoL!N~C)Lph!"
    ")BeBlj>tGr<V%0?Yxw*wr(fLfb3Su|L+iY5(~ZJvhf0$$P@46{%AtL_){=kFo;OG9G<-uW<!c=LmFr<64?(K-K>9`92YtUi^Azzf"
    "Gl~Bi8;)jfHMiMol&h+f?DI3MJtoQy-;I3-zvq+B*!HUMY+L5#^YsGeiC_7<a<W}aEcn*BXYPvs&nHg{4;_NM>o?NBxvgZAT~|r1"
    "gX>o&D<3^_%Jt+rUD?bY^5-3EeQ^f;)8HN4&pTIZfNi9Y&AE=7x#n6nbD-9dJIczjCvn`mO;|e*Yh3QhcZbG)-6rH%yT{(OO6#;A"
    "LBC<Ta6J2Led+||B&_GVflI4vdR6({<}g2i=NPO|+ca=(1=m>gSAK+r+|q>$I35urgB*Iu9|&E>jdu(aU)GJWEa;2r!g%Rcjv>#G"
    "PxclkcDQnWmNK^jVh)PA?)LduJ_C3h8UVy>xG(-lj?)c;kT(N*kqY|e!A+%VtF*@BEaeuc5vMeX98WR#d9|-y35<S<-FpP%QjqHq"
    "<xCf53hVdcT;n-N#xVCqy0Dz_^Qqba#IMLZ06c|tKfslloyGk7jSZNOCeL)8MStlgN;q%Srw$RPGD78>tvZ;>oD}fCdv5l#?0Hv`"
    "ufVzaWM6CJ4dgw5$vju&BI;J^QzN;asdEq`6`ye!G3u{3)^INKa$G>(XIsf~#!PuXAFO_VGRHS^W`M(VfBSEEKVO%#jJhtL?}oFC"
    ";8`(Nf#E$zN~FIV8V>sp(>&5#jkWy{yXF0+T==>_<3t)8hEm7kxd3M?E<Q@rGfsTZ414#(HEw5fF0%hq*k^1DZ?6jFjF?UCs<C05"
    ")!$*v@d|FsY#HL+bYU;`gGpiNyQ8hm=4ri<zV_Si$Q^lGW$pVw>P+fWN0EE-H5B`^Z-6gaRdce|uY}gE3(ErD6ypXPa*ym-v#*t9"
    "ICWIM7U>;a*8^Mwcsk764^*7Re!TH|Xv>DM?|U1L<a~_(s#P_;nEMj_wm$VM<$)Z-v7V}}W4Y6XB<Evj7V^nzYv+SFXF1*<#_{fb"
    "G1@hO&jAmWs_iKr3G2W1wtJ2d7Xplbs5s&M)xVPDKs~qZU~88FVe2a$<CAxO5yq@|E!5rq55wVq4o#mtVg2rj@WCISG~4^XVa1vk"
    "!W-~|4A~wo>HTK7=em+Ot=ns1^ATO*Kk8e>6+2qSo)Zm^4rw2oE`5PH;S#e`_I+fmddqB-*#+i-ZhU7|%#@6dSD)QGW<T2_UQ|9X"
    "{?gBftqr}l`(lUF2KC%`eWh{VO<K>MiVGM+T%Ime69c**dm?qBJJ+>#E}(<ibYqDbV5h=Y<o9e%pCD{S&*d~!ezJ-3o41jEyc>1$"
    "o-eyfII>hZt_CXR_Cw{`>rVWK7*g;Dt|b^t|D*F7z`(vff#<I3r*#Y#C<k_H`l!m9;TtZaFBEa~?!WTf+2AO)tVb*>V)Ef@-;*}E"
    "Yj?|Nn_>-eUthb2;_z3?pV?CQ-*wQ?UATYMW9WgaT`%2J@ep&R-S5VAs$37-n|fHsGr&==C>J(-?C%C&udAJG^e)us`kDZPm><m7"
    "I`7AP>)8zQI1E!+@q1g<eRtv-_*y;pTOT{GgSeU?`+K0Z-4Ly__y%*1yWedTaSHg2bYVyEJaLuE(NL+_@(#jwKXHAud~%%L>vZBO"
    "&kgoL@V?PK`Z^D@8K>a+0R~am+}N<ca!ZsjrUaP8b=O0p>jZ|1C+IG{atCR@+c5VR)@k#6Y}Vf=zZR}Jd}T;f{vk}w-xLy$|2B;N"
    ";GwW$#{=Q>m;PYen_=44d&3jQycKR4^|!EgpLTKJkDEgC{U3#g&v`FwJm??c^1nO~?(SI(rFS-mFD7-4pX|Lf-m|!ET+w%KY#P<h"
    "u=b0wcxNV-KD(P?b?c9nh`lk)-Ei11{(QvGWHV`MRU9)-C8ouEk<FkTfnS%gj%O{NB`o=))~oAH+~$1Bo{H@lEv<Kd#?!#Zj5rsM"
    "<?1CJ4miFp%;E3;+)`=y#~Su4qy7Ya-*aM?GKRx&mGaefQJl?u#iHz~ez`#Rnn1rke8Qfi51iRicHfc8budi4SA{V90O?JJGR7Qx"
    "@D^An1Kg9xEcF)W*oV1I^U2m${t5PZnr*d;Uuh|=40x20()C=UzK^kWyyBa>ikFyU*G}J+Zxk_d&Yg7=E*~dt#Ym0KVeIqiLZ$eW"
    "$<iP6RXo7X{2{~X+xJ)-@Ep4nfBSgpuQ~$<wo|t;UiIHW>!%Eto@Zw+#?gua7{GbWKhKsgznk*>&sQGEE*!JIUg|KFX>YskShi<l"
    "1NsB<%8ny$M;?cKa!-xTGR1jyQ`|r&j%knoDc4*(&CZ`rTo26_`klu-O%&f;u6enaVt_^|&Z1IrEJ@}pfF8>^5?r^p)~3Ffz6s=@"
    "^BB_06kCm)cbyd@nPm($^yjb(5yR+tv5r%`<w5i@c;4e{cI998Jl560*L;9)?<nG6*Vm3z9`?0dzuvXX2gu(von?Yu4qprOwyqmq"
    "K_7?b`a40f_23GE)?7bBn#w(i2^t%KPmf_9E!TXH*YgkI+UlOS>w@5#jd9A8bu#%6&qF_2KCB0%^*x+9M|@qi{S>==qVg^OPX3tV"
    "h%sPy{r?#7KQKR@{ZCtYkTd*vUenmHoX<lJe$S<j*!`;%xA`jd=3d7KNfV!;9^%G3YH3${t}x`@`X_VCAqLicxYKC&A}?dQ0Nb@z"
    "ZMz$_zOJWVq#V@Kh|!R*-Q!r#SDZ7nsGEt^f$`ms_JVR>RZEwj(HdSSGjCxrccY!Zi8^ce={=7BD&-42Sbd>hal^+6gJu;Eae~>U"
    "rzuzDUdnM+qudGyDL42u#SH8&4gN)3d(AcD2MO~&Y4*%A>4r1pX<&PMjO-QEoa1;tIo0a_Fs-Y7HFKExy7vY73U5>Hx;c!c-d1w6"
    "a+1v__VK<pp8ehN%`C02K9RZfd@X<<nUihDnlZ}HaG?C{Cux1YTR1M#g%iTp-z=4%d9HYh&B}N1wblD#;>UDhA@h-cvVFDs-<gUr"
    "I8*B_9H=!ZE>}$X`D{1On|HkOip?P&cm3c6!Tl<)2t(Bg|1G87-T9Sk!b69gtGp8H8DkI4GjvM6?(D71;hD{>Bezo4e5P`LqTjyG"
    "+;onu*C}s$fm(cDp9OaGpT(1{51o&@K{+0wci(95eFfJjTvhX_IF_HGFNxr9Jf}Ni%g*N7q&|+nW1KkR5<G_NTII$6Ubze|Q*MNv"
    "^B#PsPXzcw(uJ3qFQJ%wo7frg-yYj?gVn=}%DJ#yy6ck^*MK$MpH*!7Y8^*@hE2q@WzEPfiaZ0bQ=euXtg1mizuEL%Ha2VsZo}o7"
    ")6ji6Z`kK+gX^`h*8Ta!^q!w|sq#JkllyexD)x7bRdAuMv3M(BZ6oJ{5VLV3W3wDjV0>@q_{t~mrd}R;cF$=7U(^5W`L{41=Hlo6"
    "XTSfBc@v7c+m$czIptjWnY^6m0|B-G&vl>T<Y5OP_vanK*Rp~A{jA3QgPg-rC*Bs<i%ay_c+W?;FXxQLhEtf2)8iXo48E57ec}Xe"
    "X3hX`v%p2(A9q)Nk3T4P;#$RYtPZml!#28teL9=jZ1;VH@8o*@KQIOj_2F3h`rvDPePMlLweok|z&H<&k2{lb9$VHU=i<f8y@E9Z"
    "|NYbH%=J*rd2S2m%m+#@dAV}3!e+Qa{N~e&3up?yzVroLD?VL7z71fGlce8oB-TmQqF>(<Ja5o7jEzHGZY$ZTk^Vhn2JoJU7x7rK"
    "o0VJlJZb^*$z#Q7{5t$l@!pKJFHW9u$qiN2W1e{`6ra5(y#CuO!t|5u+Ie?{^13TS>ggB5yfd#4%f^2k?wNXfcq-K)X7l%l6_0EU"
    "8*ckYcw=_kxOvW>!rJX`gsPdZTbU-sZ-=)sY<~#tnPn&5Y8d>&Sl8x=`1uzxHtpFaK6UJ%cuDV?_)9+@anq}6z73A;lftlFl>4n4"
    "{WG3-zl502{dbkpt`5;TeQ6WJe_o#&sx_f%s1-;gJ`6k82uI&_UvdxXa6P{seEU6Qr|%*E`4Z)QN=jc^NqmTSVfTeqDXzG^wBwy+"
    "N0-wF=~{tKid!BcPNO|}ja2Op&bK?(^cVi@#OG{T-(A|`68mjnbU$?<hH*{s&N%cc`k25o_<N0JS&@6%=5Fa|k!Pd3Va*!G6<|%X"
    "RITSM_ZWuO(hkgLT!gRxJyq+Db>o^<?$b*$E<RoGT>g$#d>sMQKkWEbH3O9Y3OU95@5+OR>&u#ZN<-hB`Tcz@y#7{J`2H4K`4C6{"
    "ZSdS%y_DBrjM}1=@+ov9pW<tq?~I>Dp1!gFC$~XM;+<KGr*Pc^*KGGEhXaj*=R!G<oQ%IiN;nR2(bMVkMC`D~mUPgw+pBN=$iC_2"
    "C{rG@UeX9p);b4$cI9bAnGw5!K8XBp&apIeZ8%_Bw3R>mU97VdGZ#*&erk^qGnQ@bQ2p~y4~KhBy)w-E%d>{Re;YPle_L4H`sGmm"
    "@=f9AnXiNn?YD-eyT1)T|MZxR%PrxBNk50W+rJ3QU;j_|;^i+x;_UWub5%>eXX(OpoOkoVG4uO1vHXBbV)-?bW9i7-<Ay=A;;UJ!"
    "W5eCk?i>4yKi~7*wHF2+rq4sz?>ujQ895`w2ce!(ckX*_MIOtsRi$v@N7CT;P+rhZS~H-faNQ7%L*&C*sCC~y=6as>sS@he%bNR&"
    "^BF)+4qTz@w9$XNC<o<M_#CNAN!6myei>XJ&?fr-V|$Jv#J`s{|1T`*e4gS^J`0YWd+5D-NT-h&o&MC|G&W%Uy*^swbB28Q1GM(#"
    "kX`zdoxGZJdqu4YxO45$_R3X1Ksug|S`T0f=S9R3V9o+I_dNfdH2y{@cU@a?ukEz9$ypj}+k@+6x^Rt7*FFtV&V*LVliyL=hOy!{"
    "cjo=;XY*QL;{5tlE9MJL7bZ!AQNgj}@dQ5w&w<$Q|E#mnUbz59DK6-XfOS20YVr|xkS@FrP4%vI5q9R40e|D;_WMwCLrbe$U*BWP"
    "`nKfi5Lbr2QP%u+a6Y1!l>>2GEySU?h9CbC-tSt(+_#B%?u2iH=V9!kd9tH4W}U^wwB`TOPK|}nQ&oy7=)!eke7%z%X5YZy(Uraw"
    "_+lL2cF~&a-RRFi%shN2$Xn()ig3J)ImMyN2Buq81ME43e8JFnhiIL+ue1((cdg&{2z_Ub4V~zRDCXdE>aRR@_^m`fl-;?eo%?r&"
    "ntf9)E$W3@FRfI*tfbkjN$D5L!_%EcS~&*MCsoXCqUPTE4AwL-TeiD>e#Y#BuEN<>JnsJNp;}YD2g~Q{n|D^8f&I0nVI^W=gi|`$"
    "JCreA9lBSKf9S$k1msD#{~F}Pq6T%#`X17*_m@un2<f2*QDY37-Sf%8m)DMUn@_@b{<gi_9L4~+zX$%}7wxy#3`>rbuWO{V;peez"
    "DD%!7!Dt_k^Y21Gl*i+Dr_L3ceb2o;keCm7$J2#Q!sip{UtCp#_N`(&`5KOE886_zy$*~;&L>OtPCcm2_FQizS`TNE@^$}(z9;y}"
    "Z2pu#E2(v;D)}3}Z&?q$Zg;M$=xY)}cio5Y1)VOQY2%mcsv#$qx5-dGzpS}ju@T#=VaH>gIM+#6upQEcN<I&{3ebPhui!89XYWBQ"
    "j5b9***w4)2ly|2&BA@u=g;AmE|l@Ps29&i_p?3o8u3H}`A&#oz}$|q=ab08IK-}7#JCtAJDu(O=HNM?D&#BfD82L`=DzUULh#Lh"
    "OJB9?ru$Lzk9N-|Q}Snb(Re6ROb%jWdT>2bT<_}zR!ZB07>1$p`Cq6!gs{Pe*w}%7xL|#2knr)z!Pgu56Q7$;4p$rm)()$Hzerrp"
    "&NVfMP;2dd5nSN@S|c!{@jR0JQe(q@T1OGSalCgaG5WeqY2{a|u;W_;{#SgA&!nEf=N9OQ&fq%ro_7Ru-LcZ#kF<GWmhvDV$Ic!c"
    "XE+b~g~o;^a*7z=K9>$v&Wdr$`F^g}dPD9E_`23nSKxUhKV;ki)_Jl2jo|BXO}DxltGJKJnjaWb1Al?NPk+s+K^hm&Lo#deG->H~"
    "6K613>vxV<EX<vX^_ihO9&@>NBkI?;VaoqGN}OPma<vW*=T119eF%CY=T9U1EI2t|qhUY#?>+Zs&^jaoY(HCZ9`N74W6vDNXM4WJ"
    "-u9mJnZNS1kytlmn$7Q%nWqqOI>=4#T4rb@u%65}d>8)@>ZNt-<}3e2Prk3$`DDGrZ1Hl7*~cHKs8-&}S&R$wwbTyL{5YNO4}FiX"
    "5x9h02ee1vtq{|nPgWB1W-|jhZV*!i?#<UOg)MbG*B^n7z+;vAa8B`cJ!`ZU;2f>-GKKmjtXJf@68=WNyRQ|pNaOWt<*^*A|B<Cu"
    "k7sLK{H*w*`D|<Cfk6xq;(~nL)>(>`NHG^+HgmM}B-P~pe9SLaEK7y*c~t7UndA+&l~j^vMJy5e2j;xR&mAU?F*0_{$A3_+*P%Ao"
    "E+=2U_Qip*v_reNV$FX-`Gb#yWzCO=+_Zw{YMlLSXd3mo*^h;=V(HJJ!^Pi+s<U1W%T6qh%Wo`?-z~9ueRlhJ^!4AF{oN+K@p$Vv"
    ">l>_bHZdmNc{>#E93PkMyFK=N>Daiju_A76**%s&I6IDR+dq~*+bJ&p{=;AV`G^^BY`C3vFnrtKGhK(h(Z=wH$_WhrF?5;VGOs55"
    "xW1nHg8?!2Wz9RW{I&p}d|lHT`J(?DTyI_@d#n?&pRW-<*zhN8Sj;QixW-&v6V|=@hm~iC`X{h_MscZw?K;@`-L<|~FZ$){Q}9Dn"
    "(Z=4f<}>Bmgum>G;65efB;2NT2*H1yFCS(J^|Efq!*{vEuKy4GCDH67;au39d3)w(_^&k^^;P;K+;)d<9`%Dfqd}OyLb#!w*0Flk"
    "<^%YyfHhr*pQ`O)_koT3X>d$mMq3=(1mrecRr5hWtgZ6~U8onxC%ZEqV_Qi|Ypa#fPwDy65u5y$%J-SgP5)M&FZd^4wz6U!#m~9s"
    "n&ak=wU*Ue%*_FfqQ?gyZnjo_$WK{c^{M87xLmiLzb6Oe*zWz{dH6f<`Rg{J%*f5P4eO5x*S%)ra$DHh?yh-0fVR8yfzS&a&b1;M"
    "8-RyAmpIn)ZftlFF|W+u;Ohi}51GvT2C%1H7t)n@(Q^=%$S=A*crL?{%7cjfXm4wc_*TkM(4D#m=d$*UU(`Jqu6^Pkp}fam4R7BZ"
    "?)mehVRib2aMq{xj)xUP)niTJ;d5HrxcOg5J^gK1{{7&%^u=y*>!yyeX;hnd(Vr1hWA%F2{4nz91LN||KZU!etcaOf8*48czCL<E"
    "+`8zG@sjBW#N`JMwEGN?Rll_{QQ0l7PWSx9pYJ@{RQ5gKr25o!>OhLQ<>CZJus>t|_cbz;*3W?bMk(J~7sh|Ue*%B5_oWJLCB#@*"
    "^Um?+kM=*bm|7H%75JJtUH^YR${x}4#P*K9hSn7N*};+d+8jyZSC3Jj!Fxh`1Km2}Q?TwF{Nc_qwb!~6<r>HRsdcGqI$D?*_!Mi%"
    "brM&2IQt*CHsEg0NzlUT1F=eF>c4%oW@Rg3{~rS4=~J~qnuMQ&`@Bn(OSFUd59H<OK;9DCnsuA7hDZhT6X5>N>7^J4?>QE~wR$RN"
    "{y`jb{Xrk)>F>aJ3FK(VX69?Xfvy~L=T7)aoPA&E7%*nPvGSEGCuv*d>nzv0gr6xUs2}4SeEotc92>y*Uf(^H`yalLQm)gA`f(1i"
    "QtOoTQTw%Kj=si*d3<-25xOpq59urad`s%0oKGDm9YcTm<h?(26JFn$+hDlfw@f@oSGCdC<Ud^N(8ua==Q<1DaBf15Mqi`Q*H3Xj"
    "SSRIG>>Ot;?!$Qlxl*&4Z-s5(SKCDU54n`{$%~kWs<Gj1`+O&lqaNy0$UhA2)hE=9`+8;flH1NFk>eWqxLVm<{(IZMru9kRq^|}3"
    "KKMA$Hi^Vvm|M)(lYPmK-A#^s-KMXt{U5aBf2LL+|1;?8SmVxfNIzz6`W|s5)^O@M5qXdP#hg%x9l*7zTEw;eXd8GLJnJ?4pWY@P"
    "-uw3XSv$YUe)|LSb~wlLv2rVaVDESwIW6EJ<UvE;Y}c*7q`c4#<TX4V0mokq-nP#K&wF{l)?NcY{}TO}SgSTwdk5ESLp!8u|H}32"
    ")@^!2KE(gXH?@)Tem+^SYd*9xZe>|hweZtjuiWYXv3DpKX8Uh|U#&iM9_zr@#oH2`*S^v=j8$-Rn+$g##|`qdp<kY1Wk;ODt6E16"
    "w#f6gy~%PTw?w+|uHEy$S{HYdJ>xI3Z~huq)x1NWIed`W%r|ykL3zL6&-#{ai$3RZ8Bf!`$7kS>#m_xp|BXe)NUYn`Wcz>1PyLp1"
    "U~jW$-$m?MpL)v9-L9PDA6a|6VV@rjZd-k2bpoBi|CGP{&4BtxxjgSSxW_-rjy{yXyHD~_d-w0P7Qo+`htD|*<Y8-4e(&po>nk6n"
    "&A<4$rx5=i{`5Mu7LQ2plAs2&thtsp8`fSe=I&!(f$f2{Rgv$**C712mGx08Ym;sNwa>sz*Rd?f>AqtPu)!C69&&s;2MN0p{oy5Y"
    "LAZxw@muJFh5yv^es86|z+=JhWG;T>SWnfiP}_b;jObk4?aF_+-0J3Ww*LbaAKNweNGtKA^zFCVJLS37JNh;9p}Bt@bNbctqkm*="
    "@)`5Qmo>w03eNmG+E$)>{<7eHqJ8c9`-8`^Znv_*ru?H}kk<mn2J+HD%hma~rS{)|IGN8C_ki*9C*r#L)Zz456mwe`bK&veQyKpN"
    "|8---1&oVvjp=e}U-0~*a@eD9*4r~b56;i-#T<v|^XR*t^Kpq{F7_wi1uZPr-|%&tM^aaUI0>w=hjj{4wGXntrVD?@`hnto*Rzae"
    "%@_mkS-W0M?S^Y1PGDQPfAK`|lYb(P^LYnY`)zyfJ66u;ZU3fpHt^lw8&Egy??xWhR?I`>T*n)1#~o|#l)mCpYBunHhdv3cpDx_P"
    "y7F~7;6M0);{rZ**O;xBCgjfn>uMnvhqvk5iX#W72u)Bw`oDaQ*8|91rVCH-e*=6HG$*bH+Nv0u4Z%5VjMu*hX#QO9zFupGme~K}"
    "T>I@Y_HXDY@PFXz0X$~oaSQvVuWNY&`zW;V`6Ql=aXU+XgFlLwz2*N6m{z)>W1%3we*>RE0jXC{fyc9VELaW>#~lkmbrYzLybZnM"
    ")7S=5ZqEbHv4hqXgYMOX+`Bpn9DkrOwU~egXuEu)4P@Q|)V3-Frw`CxI>@|2A&~w9y`MG)oZcbfaTYvw3A*DLRPKQKMKJ->q339V"
    "`Z_y+;r<)ECIWOOA;>+TwcEfnY<)o#l;=J}&nh?p9gEfnkAH2Jc?doW2(o7FA-If=3D^TZQ}lx99_V?JYk=zZ0n5`<;IlJ!EI4KZ"
    "03BmmDF"
)


def align_up(val: int, align: int = ALIGN) -> int:
    return (val + align - 1) // align * align


def set_ext_data(tensor: TensorProto, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> None:
    tensor.data_type = dtype
    del tensor.dims[:]
    tensor.dims.extend(dims)
    tensor.ClearField("raw_data")
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in (("location", location), ("offset", str(offset)), ("length", str(length))):
        e = tensor.external_data.add()
        e.key, e.value = k, v


def make_ext_init(name: str, location: str, offset: int, length: int, dtype: int, dims: list[int]) -> TensorProto:
    t = TensorProto()
    t.name = name
    set_ext_data(t, location, offset, length, dtype, dims)
    return t


def make_bf16_tensor(name: str, u16_arr: np.ndarray, dims: list[int]) -> TensorProto:
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.BFLOAT16
    t.dims.extend(dims)
    t.raw_data = np.ascontiguousarray(u16_arr, dtype=np.uint16).tobytes()
    return t


def load_calibrated_kv_scales(kv_scale_file: Path | None = None) -> dict[str, np.ndarray]:
    """Return `{'node_GuardedDSA_{4..15}_{k,v}_scale': float32[512]}` from `kv_scale_file` or embedded defaults."""
    if kv_scale_file is not None:
        data = json.loads(kv_scale_file.read_text())
        layer_ids = [int(x) for x in data["layer_ids"]]
        k_map = {
            lid: np.asarray(s, dtype=np.float32) for lid, s in zip(layer_ids, data["scales"]["k_scales"], strict=True)
        }
        v_map = {
            lid: np.asarray(s, dtype=np.float32) for lid, s in zip(layer_ids, data["scales"]["v_scales"], strict=True)
        }
        out: dict[str, np.ndarray] = {}
        for idx, lid in enumerate(range(3, 48, 4), start=4):
            out[f"node_GuardedDSA_{idx}_k_scale"] = k_map[lid]
            out[f"node_GuardedDSA_{idx}_v_scale"] = v_map[lid]
        return out

    raw = zlib.decompress(base64.b85decode(_DEFAULT_CALIBRATED_KV_SCALES_B85))
    arr = np.frombuffer(raw, dtype=np.float32).reshape(24, 512).copy()
    out = {}
    for k in range(12):
        dsa_idx = 4 + k
        out[f"node_GuardedDSA_{dsa_idx}_k_scale"] = arr[2 * k]
        out[f"node_GuardedDSA_{dsa_idx}_v_scale"] = arr[2 * k + 1]
    return out


class BinaryBlobWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rel_name = path.name
        self.fh = open(path, "wb")  # noqa: SIM115
        self.pos = 0

    def write_aligned(self, raw: bytes, dtype: int, dims: list[int], align: int = ALIGN) -> dict:
        aligned = align_up(self.pos, align) if align > 1 else self.pos
        if aligned > self.pos:
            self.fh.write(b"\x00" * (aligned - self.pos))
            self.pos = aligned
        off = self.pos
        self.fh.write(raw)
        self.pos += len(raw)
        return {
            "location": self.rel_name,
            "offset": off,
            "length": len(raw),
            "dtype": dtype,
            "dims": list(dims),
        }

    def close(self) -> None:
        self.fh.close()


def layer_base_node_id(layer_idx: int) -> int:
    """Return the canonical ONNX node-ID base for `layer_idx` in `0..47`."""
    base = 87
    for i in range(1, layer_idx + 1):
        if i == 1:
            base += 149
        else:
            base += 147 if ((i - 1) % 4 == 3) else 109
    return base


def build_embedding_model_proto(vocab_size: int = 248320, hidden_size: int = 2560) -> onnx.ModelProto:
    """Procedurally construct `embedding.onnx` (per-row INT8 Gather + FP32 scale dequant -> BF16)."""
    i8_bytes = vocab_size * hidden_size
    s_bytes = vocab_size * 4
    inits = [
        make_ext_init(
            "model.embed_tokens.weight", "embedding.onnx.data", 0, i8_bytes, TensorProto.INT8, [vocab_size, hidden_size]
        ),
        make_ext_init(
            "model.embed_tokens.scales", "embedding.onnx.data", i8_bytes, s_bytes, TensorProto.FLOAT, [vocab_size, 1]
        ),
    ]
    nodes = [
        helper.make_node(
            "Gather", ["model.embed_tokens.weight", "input_ids"], ["embed_i8"], name="Gather_weight", axis=0
        ),
        helper.make_node("Cast", ["embed_i8"], ["embed_f32"], name="Cast_weight_f32", to=TensorProto.FLOAT),
        helper.make_node(
            "Gather", ["model.embed_tokens.scales", "input_ids"], ["scale_f32"], name="Gather_scale", axis=0
        ),
        helper.make_node("Mul", ["embed_f32", "scale_f32"], ["out_f32"], name="Mul_dequant"),
        helper.make_node("Cast", ["out_f32"], ["inputs_embeds"], name="Cast_out_bf16", to=TensorProto.BFLOAT16),
    ]
    graph = helper.make_graph(
        nodes,
        "int8_embedding_graph",
        [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_tokens"])],
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.BFLOAT16, ["num_tokens", hidden_size])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    return model


def build_dflash2_model_proto(
    tensor_locations: dict[str, dict],
    base_kernels_u16: dict[str, np.ndarray],
    *,
    block_size: int = 8,
    num_draft_tokens: int = 7,
    selector_top_k: int = 16,
    sliding_window: int = 2048,
) -> onnx.ModelProto:
    """Procedurally construct the 310-node `dflash2.onnx` block-diffusion speculative drafter graph."""
    eps = 9.999999974752427e-07

    def make_const(name: str, arr) -> onnx.NodeProto:
        return helper.make_node(
            "Constant",
            [],
            [name],
            name=f"{name}/Constant",
            value=numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name),
        )

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Cast",
            ["aux_hidden_states"],
            ["/dflash2/aux/Cast/output_0"],
            name="/dflash2/aux/Cast",
            to=TensorProto.BFLOAT16,
        ),
        helper.make_node(
            "MatMulNBits",
            ["/dflash2/aux/Cast/output_0", "dflash2.fc.MatMul.weight.qweight", "dflash2.fc.MatMul.weight.scales"],
            ["/dflash2/fc/MatMul/output_0"],
            name="/dflash2/fc/MatMul",
            domain="com.microsoft",
            K=12800,
            N=2560,
            accuracy_level=0,
            bits=4,
            block_size=32,
            weight_prepacked=1,
        ),
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["/dflash2/fc/MatMul/output_0", "dflash2.hidden_norm.weight"],
            ["/dflash2/hidden_norm/output_0"],
            name="/dflash2/hidden_norm",
            axis=-1,
            epsilon=eps,
            stash_type=1,
        ),
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight", "input_ids"],
            ["/dflash2/embed_tokens/Gather_i8/output_0"],
            name="/dflash2/embed_tokens/Gather_i8",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["model.embed_tokens.weight.row_scales", "input_ids"],
            ["/dflash2/embed_tokens/Gather_scale/output_0"],
            name="/dflash2/embed_tokens/Gather_scale",
            axis=0,
        ),
        helper.make_node(
            "Cast",
            ["/dflash2/embed_tokens/Gather_i8/output_0"],
            ["/dflash2/embed_tokens/Cast_f32/output_0"],
            name="/dflash2/embed_tokens/Cast_f32",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Mul",
            ["/dflash2/embed_tokens/Cast_f32/output_0", "/dflash2/embed_tokens/Gather_scale/output_0"],
            ["/dflash2/embed_tokens/Mul_dequant/output_0"],
            name="/dflash2/embed_tokens/Mul_dequant",
        ),
        helper.make_node(
            "Cast",
            ["/dflash2/embed_tokens/Mul_dequant/output_0"],
            ["/dflash2/embed_tokens/Cast/output_0"],
            name="/dflash2/embed_tokens/Cast",
            to=TensorProto.BFLOAT16,
        ),
        make_const("dflash2.const.0", 0),
        make_const("dflash2.const.1", 1),
        make_const("dflash2.const.2", block_size),
        make_const("dflash2.const.3", [-1, 1]),
    ]

    def emit_conv_prepare(lid: int, kind: str, norm_out: str) -> tuple[str, str, str]:
        pfx = f"/dflash2/layers.{lid}/{kind}"
        mm_out = f"{pfx}/kernel_projection/MatMul/output_0"
        r4d = f"{pfx}/kernel_projection/Reshape4D_out"
        add4d = f"{pfx}/kernel_projection/AddAll_out"
        masked4d = f"{pfx}/kernel_projection/MaskTap1_out"
        r3d = f"{pfx}/kernel_projection/Reshape3D_out"
        s0, s1, s2, s3 = [f"{pfx}/kernel_projection/SplitAll_{k}" for k in range(4)]
        c_p0 = f"{pfx}/prepare/coef0/Reshape/output_0"
        c_p1 = f"{pfx}/prepare/coef1/Reshape/output_0"
        c_f0 = f"{pfx}/finish/coef0/Reshape/output_0"
        c_f1 = f"{pfx}/finish/coef1/Reshape/output_0"
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [norm_out, f"dflash2.layers.{lid}.{kind}.kernel_projection.MatMul.weight"],
                    [mm_out],
                    name=f"{pfx}/kernel_projection/MatMul",
                ),
                helper.make_node(
                    "Reshape", [mm_out, "dflash2.shape_m1_4_160_1"], [r4d], name=f"{pfx}/kernel_projection/Reshape4D"
                ),
                helper.make_node(
                    "Add", [f"{pfx}/base_kernel_all_1x4x160x16", r4d], [add4d], name=f"{pfx}/kernel_projection/AddAll"
                ),
                helper.make_node(
                    "Mul",
                    [add4d, "dflash2.conv_tap_mask_1x8x4x1x1"],
                    [masked4d],
                    name=f"{pfx}/kernel_projection/MaskTap1",
                ),
                helper.make_node(
                    "Reshape", [masked4d, "dflash2.shape_m1_4_2560"], [r3d], name=f"{pfx}/kernel_projection/Reshape3D"
                ),
                helper.make_node(
                    "Split",
                    [r3d, "dflash2.split_4x1"],
                    [s0, s1, s2, s3],
                    name=f"{pfx}/kernel_projection/SplitAll",
                    axis=1,
                ),
                helper.make_node(
                    "Reshape", [s0, "dflash2.shape_m1_2560"], [c_p0], name=f"{pfx}/prepare/coef0/Reshape2D"
                ),
                helper.make_node(
                    "Reshape", [s1, "dflash2.shape_m1_2560"], [c_p1], name=f"{pfx}/prepare/coef1/Reshape2D"
                ),
                helper.make_node(
                    "Reshape", [s2, "dflash2.shape_m1_2560"], [c_f0], name=f"{pfx}/finish/coef0/Reshape2D"
                ),
                helper.make_node(
                    "Reshape", [s3, "dflash2.shape_m1_2560"], [c_f1], name=f"{pfx}/finish/coef1/Reshape2D"
                ),
            ]
        )
        if lid == 0 and kind == "attention_conv":
            nodes.extend(
                [
                    make_const("dflash2.const.4", [-1, 4, 160]),
                    make_const("dflash2.const.5", [-1, 160, 1]),
                    make_const("dflash2.const.6", [-1, 2560]),
                ]
            )
        nodes.append(
            helper.make_node(
                "Mul", [norm_out, c_p0], [f"{pfx}/prepare/term0/Mul/output_0"], name=f"{pfx}/prepare/term0/Mul"
            )
        )
        if lid == 0 and kind == "attention_conv":
            nodes.extend(
                [
                    make_const("dflash2.const.7", [0]),
                    make_const("dflash2.const.8", [1]),
                    make_const("dflash2.const.9", [-1]),
                ]
            )
        nodes.extend(
            [
                helper.make_node(
                    "Slice",
                    [norm_out, "dflash2.const.7", "dflash2.const.9", "dflash2.const.7"],
                    [f"{pfx}/prepare/shift1/Slice/output_0"],
                    name=f"{pfx}/prepare/shift1/Slice",
                ),
                helper.make_node(
                    "Concat",
                    ["dflash2.zero_row_1x2560", f"{pfx}/prepare/shift1/Slice/output_0"],
                    [f"{pfx}/prepare/shift1/Concat/output_0"],
                    name=f"{pfx}/prepare/shift1/Concat",
                    axis=0,
                ),
                helper.make_node(
                    "Mul",
                    [f"{pfx}/prepare/shift1/Concat/output_0", c_p1],
                    [f"{pfx}/prepare/term1/Mul/output_0"],
                    name=f"{pfx}/prepare/term1/Mul",
                ),
                helper.make_node(
                    "Add",
                    [f"{pfx}/prepare/term0/Mul/output_0", f"{pfx}/prepare/term1/Mul/output_0"],
                    [f"{pfx}/prepare/sum1/Add/output_0"],
                    name=f"{pfx}/prepare/sum1/Add",
                ),
            ]
        )
        return f"{pfx}/prepare/sum1/Add/output_0", c_f0, c_f1

    def emit_conv_finish(lid: int, kind: str, body_out: str, c_f0: str, c_f1: str) -> str:
        pfx = f"/dflash2/layers.{lid}/{kind}"
        nodes.extend(
            [
                helper.make_node(
                    "Mul", [body_out, c_f0], [f"{pfx}/finish/term0/Mul/output_0"], name=f"{pfx}/finish/term0/Mul"
                ),
                helper.make_node(
                    "Slice",
                    [body_out, "dflash2.const.7", "dflash2.const.9", "dflash2.const.7"],
                    [f"{pfx}/finish/shift1/Slice/output_0"],
                    name=f"{pfx}/finish/shift1/Slice",
                ),
                helper.make_node(
                    "Concat",
                    ["dflash2.zero_row_1x2560", f"{pfx}/finish/shift1/Slice/output_0"],
                    [f"{pfx}/finish/shift1/Concat/output_0"],
                    name=f"{pfx}/finish/shift1/Concat",
                    axis=0,
                ),
                helper.make_node(
                    "Mul",
                    [f"{pfx}/finish/shift1/Concat/output_0", c_f1],
                    [f"{pfx}/finish/term1/Mul/output_0"],
                    name=f"{pfx}/finish/term1/Mul",
                ),
                helper.make_node(
                    "Add",
                    [f"{pfx}/finish/term0/Mul/output_0", f"{pfx}/finish/term1/Mul/output_0"],
                    [f"{pfx}/finish/sum1/Add/output_0"],
                    name=f"{pfx}/finish/sum1/Add",
                ),
            ]
        )
        return f"{pfx}/finish/sum1/Add/output_0"

    prev_res = "/dflash2/embed_tokens/Cast/output_0"
    prev_mlp_sum = ""
    for lid in range(5):
        in_norm = f"/dflash2/layers.{lid}/input_layernorm/output_0"
        if lid == 0:
            nodes.append(
                helper.make_node(
                    "SimplifiedLayerNormalization",
                    [prev_res, "dflash2.layers.0.input_layernorm.weight"],
                    [in_norm],
                    name="/dflash2/layers.0/input_layernorm",
                    axis=-1,
                    epsilon=eps,
                    stash_type=1,
                )
            )
            attn_res_base = prev_res
        else:
            attn_res_base = f"/dflash2/layers.{lid}/input_layernorm/output_3"
            nodes.append(
                helper.make_node(
                    "SkipSimplifiedLayerNormalization",
                    [prev_res, prev_mlp_sum, f"dflash2.layers.{lid}.input_layernorm.weight"],
                    [in_norm, "", "", attn_res_base],
                    name=f"/dflash2/layers.{lid}/input_layernorm",
                    domain="com.microsoft",
                    epsilon=eps,
                )
            )

        prep_attn, cf0, cf1 = emit_conv_prepare(lid, "attention_conv", in_norm)
        nodes.extend(
            [
                helper.make_node(
                    "Concat",
                    [prep_attn, "/dflash2/hidden_norm/output_0"],
                    [f"/dflash2/layers.{lid}/attn/qkv_rows/Concat/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/qkv_rows/Concat",
                    axis=0,
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"/dflash2/layers.{lid}/attn/qkv_rows/Concat/output_0",
                        f"dflash2.layers.{lid}.attn.qkv_proj.MatMul.weight.qweight",
                        f"dflash2.layers.{lid}.attn.qkv_proj.MatMul.weight.scales",
                    ],
                    [f"/dflash2/layers.{lid}/attn/qkv_proj/MatMul/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/qkv_proj/MatMul",
                    domain="com.microsoft",
                    K=2560,
                    N=6144,
                    accuracy_level=0,
                    bits=4,
                    block_size=32,
                    weight_prepacked=1,
                ),
                helper.make_node(
                    "Gather",
                    [f"/dflash2/layers.{lid}/attn/qkv_proj/MatMul/output_0", "qkv_row_map"],
                    [f"/dflash2/layers.{lid}/attn/qkv_gather/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/qkv_gather",
                ),
                helper.make_node(
                    "PagedAttention",
                    [
                        f"/dflash2/layers.{lid}/attn/qkv_gather/output_0",
                        "",
                        "",
                        f"past_key_values.{lid}.key",
                        f"past_key_values.{lid}.value",
                        "cumulative_sequence_lengths",
                        "past_sequence_lengths",
                        "block_table",
                        "dflash2.cos_cache",
                        "dflash2.sin_cache",
                        "",
                        "",
                        f"dflash2.layers.{lid}.self_attn.q_norm.weight",
                        f"dflash2.layers.{lid}.self_attn.k_norm.weight",
                        "",
                        "",
                        "attention_metadata",
                    ],
                    [
                        f"/dflash2/layers.{lid}/attn/PagedAttention/output_0",
                        f"present.{lid}.key",
                        f"present.{lid}.value",
                    ],
                    name=f"/dflash2/layers.{lid}/attn/PagedAttention",
                    domain="com.microsoft",
                    num_heads=32,
                    kv_num_heads=8,
                    local_window_size=sliding_window,
                    is_causal=0,
                    do_rotary=1,
                    rotary_interleaved=0,
                    qk_norm_epsilon=eps,
                ),
                helper.make_node(
                    "Gather",
                    [f"/dflash2/layers.{lid}/attn/PagedAttention/output_0", "block_row_index"],
                    [f"/dflash2/layers.{lid}/attn/out_gather/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/out_gather",
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"/dflash2/layers.{lid}/attn/out_gather/output_0",
                        f"dflash2.layers.{lid}.attn.o_proj.MatMul.weight.qweight",
                        f"dflash2.layers.{lid}.attn.o_proj.MatMul.weight.scales",
                    ],
                    [f"/dflash2/layers.{lid}/attn/o_proj/MatMul/output_0"],
                    name=f"/dflash2/layers.{lid}/attn/o_proj/MatMul",
                    domain="com.microsoft",
                    K=4096,
                    N=2560,
                    accuracy_level=0,
                    bits=4,
                    block_size=32,
                    weight_prepacked=1,
                ),
            ]
        )
        attn_sum = emit_conv_finish(
            lid, "attention_conv", f"/dflash2/layers.{lid}/attn/o_proj/MatMul/output_0", cf0, cf1
        )
        post_norm = f"/dflash2/layers.{lid}/post_attention_layernorm/output_0"
        post_res = f"/dflash2/layers.{lid}/post_attention_layernorm/output_3"
        nodes.append(
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                [attn_res_base, attn_sum, f"dflash2.layers.{lid}.post_attention_layernorm.weight"],
                [post_norm, "", "", post_res],
                name=f"/dflash2/layers.{lid}/post_attention_layernorm",
                domain="com.microsoft",
                epsilon=eps,
            )
        )

        prep_mlp, m_cf0, m_cf1 = emit_conv_prepare(lid, "mlp_conv", post_norm)
        nodes.append(
            helper.make_node(
                "MatMulNBits",
                [
                    prep_mlp,
                    f"dflash2.layers.{lid}.mlp.gate_up_proj.MatMul.weight.qweight",
                    f"dflash2.layers.{lid}.mlp.gate_up_proj.MatMul.weight.scales",
                ],
                [f"/dflash2/layers.{lid}/mlp/gate_up_proj/MatMul/output_0"],
                name=f"/dflash2/layers.{lid}/mlp/gate_up_proj/MatMul",
                domain="com.microsoft",
                K=2560,
                N=15360,
                accuracy_level=0,
                bits=4,
                block_size=32,
                weight_prepacked=1,
            )
        )
        if lid == 0:
            nodes.append(make_const("dflash2.const.10", [7680, 7680]))
        nodes.extend(
            [
                helper.make_node(
                    "SwiGLU",
                    [f"/dflash2/layers.{lid}/mlp/gate_up_proj/MatMul/output_0"],
                    [f"/dflash2/layers.{lid}/mlp/act/MulUp/output_0"],
                    name=f"/dflash2/layers.{lid}/mlp/act/SwiGLU",
                    domain="com.microsoft",
                ),
                helper.make_node(
                    "MatMulNBits",
                    [
                        f"/dflash2/layers.{lid}/mlp/act/MulUp/output_0",
                        f"dflash2.layers.{lid}.mlp.down_proj.MatMul.weight.qweight",
                        f"dflash2.layers.{lid}.mlp.down_proj.MatMul.weight.scales",
                    ],
                    [f"/dflash2/layers.{lid}/mlp/down_proj/MatMul/output_0"],
                    name=f"/dflash2/layers.{lid}/mlp/down_proj/MatMul",
                    domain="com.microsoft",
                    K=7680,
                    N=2560,
                    accuracy_level=0,
                    bits=4,
                    block_size=32,
                    weight_prepacked=1,
                ),
            ]
        )
        prev_mlp_sum = emit_conv_finish(
            lid, "mlp_conv", f"/dflash2/layers.{lid}/mlp/down_proj/MatMul/output_0", m_cf0, m_cf1
        )
        prev_res = post_res

    nodes.extend(
        [
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                [prev_res, prev_mlp_sum, "dflash2.norm.weight"],
                ["/dflash2/norm/output_0"],
                name="/dflash2/norm",
                domain="com.microsoft",
                epsilon=eps,
            ),
            make_const("dflash2.const.11", [-1, block_size, 2560]),
            helper.make_node(
                "Reshape",
                ["/dflash2/norm/output_0", "dflash2.const.11"],
                ["/dflash2/select/Reshape/output_0"],
                name="/dflash2/select/Reshape",
            ),
            make_const("dflash2.const.12", [block_size]),
            helper.make_node(
                "Slice",
                ["/dflash2/select/Reshape/output_0", "dflash2.const.8", "dflash2.const.12", "dflash2.const.8"],
                ["/dflash2/select/Slice/output_0"],
                name="/dflash2/select/Slice",
            ),
            helper.make_node(
                "Reshape",
                ["/dflash2/select/Slice/output_0", "dflash2.const.6"],
                ["/dflash2/select/Flatten/output_0"],
                name="/dflash2/select/Flatten",
            ),
            helper.make_node(
                "MatMulNBits",
                ["/dflash2/select/Flatten/output_0", "lm_head.weight_t.qweight", "lm_head.weight_t.scales"],
                ["/lm_head/MatMul/output_0"],
                name="/lm_head/MatMul",
                domain="com.microsoft",
                K=2560,
                N=248320,
                accuracy_level=0,
                bits=8,
                block_size=32,
                weight_prepacked=1,
            ),
            helper.make_node(
                "Cast",
                ["/lm_head/MatMul/output_0"],
                ["/dflash2/topk/Cast/output_0"],
                name="/dflash2/topk/Cast",
                to=TensorProto.FLOAT,
            ),
            make_const("dflash2.const.13", [selector_top_k]),
            helper.make_node(
                "TopK",
                ["/dflash2/topk/Cast/output_0", "dflash2.const.13"],
                ["/dflash2/topk/values", "/dflash2/topk/indices"],
                name="/dflash2/topk/TopK",
                axis=-1,
                largest=1,
                sorted=1,
            ),
            make_const("dflash2.const.14", [-1, num_draft_tokens, selector_top_k]),
            helper.make_node(
                "Reshape",
                ["/dflash2/topk/indices", "dflash2.const.14"],
                ["/dflash2/topk/cand/output_0"],
                name="/dflash2/topk/cand",
            ),
            helper.make_node(
                "Reshape",
                ["/dflash2/topk/values", "dflash2.const.14"],
                ["/dflash2/topk/unary/output_0"],
                name="/dflash2/topk/unary",
            ),
            helper.make_node(
                "MatMul",
                ["/dflash2/select/Flatten/output_0", "dflash2.selector.hidden_projection.MatMul.weight"],
                ["/dflash2/selector/hidden_projection/MatMul/output_0"],
                name="/dflash2/selector/hidden_projection/MatMul",
            ),
            make_const("dflash2.const.15", [-1, num_draft_tokens, 1, 256]),
            helper.make_node(
                "Reshape",
                ["/dflash2/selector/hidden_projection/MatMul/output_0", "dflash2.const.15"],
                ["/dflash2/selector/hp/output_0"],
                name="/dflash2/selector/hp",
            ),
            make_const("dflash2.const.16", [-1, block_size]),
            helper.make_node(
                "Reshape",
                ["input_ids", "dflash2.const.16"],
                ["/dflash2/selector/ids/output_0"],
                name="/dflash2/selector/ids",
            ),
            helper.make_node(
                "Slice",
                ["/dflash2/selector/ids/output_0", "dflash2.const.7", "dflash2.const.8", "dflash2.const.8"],
                ["/dflash2/selector/anchor/output_0"],
                name="/dflash2/selector/anchor",
            ),
            make_const("dflash2.const.17", [-1, 1, 1]),
            helper.make_node(
                "Reshape",
                ["/dflash2/selector/anchor/output_0", "dflash2.const.17"],
                ["/dflash2/selector/anchor3/output_0"],
                name="/dflash2/selector/anchor3",
            ),
            make_const("dflash2.const.18", [1, 1, selector_top_k]),
            helper.make_node(
                "Tile",
                ["/dflash2/selector/anchor3/output_0", "dflash2.const.18"],
                ["/dflash2/selector/anchor_tile/output_0"],
                name="/dflash2/selector/anchor_tile",
            ),
            make_const("dflash2.const.19", [num_draft_tokens - 1]),
            helper.make_node(
                "Slice",
                ["/dflash2/topk/cand/output_0", "dflash2.const.7", "dflash2.const.19", "dflash2.const.8"],
                ["/dflash2/selector/prev_tail/output_0"],
                name="/dflash2/selector/prev_tail",
            ),
            helper.make_node(
                "Concat",
                ["/dflash2/selector/anchor_tile/output_0", "/dflash2/selector/prev_tail/output_0"],
                ["/dflash2/selector/prev/output_0"],
                name="/dflash2/selector/prev",
                axis=1,
            ),
            helper.make_node(
                "Gather",
                ["dflash2.candidate_selector.predecessor_codebook", "/dflash2/selector/prev/output_0"],
                ["/dflash2/selector/pred_gather/output_0"],
                name="/dflash2/selector/pred_gather",
            ),
            helper.make_node(
                "Gather",
                ["dflash2.candidate_selector.successor_codebook", "/dflash2/topk/cand/output_0"],
                ["/dflash2/selector/succ_gather/output_0"],
                name="/dflash2/selector/succ_gather",
            ),
            helper.make_node(
                "Mul",
                ["/dflash2/selector/pred_gather/output_0", "/dflash2/selector/hp/output_0"],
                ["/dflash2/selector/pw/output_0"],
                name="/dflash2/selector/pw",
            ),
            helper.make_node(
                "Transpose",
                ["/dflash2/selector/succ_gather/output_0"],
                ["/dflash2/selector/succ_t/output_0"],
                name="/dflash2/selector/succ_t",
                perm=[0, 1, 3, 2],
            ),
            helper.make_node(
                "MatMul",
                ["/dflash2/selector/pw/output_0", "/dflash2/selector/succ_t/output_0"],
                ["/dflash2/selector/pair/output_0"],
                name="/dflash2/selector/pair",
            ),
            helper.make_node(
                "Cast",
                ["/dflash2/selector/pair/output_0"],
                ["/dflash2/selector/pair_cast/output_0"],
                name="/dflash2/selector/pair_cast",
                to=TensorProto.FLOAT,
            ),
            make_const("dflash2.const.20", [-1, num_draft_tokens, 1, selector_top_k]),
            helper.make_node(
                "Reshape",
                ["/dflash2/topk/unary/output_0", "dflash2.const.20"],
                ["/dflash2/selector/unary4/output_0"],
                name="/dflash2/selector/unary4",
            ),
            helper.make_node(
                "Add",
                ["/dflash2/selector/pair_cast/output_0", "/dflash2/selector/unary4/output_0"],
                ["draft_scores"],
                name="/dflash2/selector/scores",
            ),
            helper.make_node(
                "Cast",
                ["/dflash2/topk/cand/output_0"],
                ["draft_candidate_ids"],
                name="/dflash2/selector/cand_cast",
                to=TensorProto.INT32,
            ),
        ]
    )

    inputs = [
        helper.make_tensor_value_info("aux_hidden_states", TensorProto.BFLOAT16, ["num_ctx", 12800]),
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["num_block"]),
        helper.make_tensor_value_info("qkv_row_map", TensorProto.INT32, ["num_tokens"]),
        helper.make_tensor_value_info("block_row_index", TensorProto.INT32, ["num_block"]),
        helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_size + 1"]),
        helper.make_tensor_value_info("past_sequence_lengths", TensorProto.INT32, ["batch_size"]),
        helper.make_tensor_value_info("block_table", TensorProto.INT32, ["batch_size", "max_num_blocks"]),
        helper.make_tensor_value_info("attention_metadata", TensorProto.INT32, [3]),
    ]
    for lid in range(5):
        inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{lid}.key", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128]
            )
        )
        inputs.append(
            helper.make_tensor_value_info(
                f"past_key_values.{lid}.value", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128]
            )
        )

    outputs = [
        helper.make_tensor_value_info(
            "draft_candidate_ids", TensorProto.INT32, ["batch_size", num_draft_tokens, selector_top_k]
        ),
        helper.make_tensor_value_info(
            "draft_scores", TensorProto.FLOAT, ["batch_size", num_draft_tokens, selector_top_k, selector_top_k]
        ),
    ]
    for lid in range(5):
        outputs.append(
            helper.make_tensor_value_info(f"present.{lid}.key", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128])
        )
        outputs.append(
            helper.make_tensor_value_info(f"present.{lid}.value", TensorProto.BFLOAT16, ["num_blocks", 128, 8, 128])
        )

    mask_f32 = np.ones((1, block_size, 4, 1, 1), dtype=np.float32)
    mask_f32[0, 0, 1, 0, 0] = 0.0
    mask_f32[0, 0, 3, 0, 0] = 0.0
    mask_u16 = torch.from_numpy(mask_f32).to(torch.bfloat16).view(torch.uint16).numpy()

    ext_names = sorted(
        k for k in tensor_locations if k.startswith(("dflash2.", "/dflash2/", "lm_head.", "model.embed_tokens."))
    )
    inits: list[TensorProto] = []
    for name in ext_names:
        if name == "dflash2.conv_mask_f32":
            inits.append(
                numpy_helper.from_array(
                    np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32), name="dflash2.conv_mask_f32"
                )
            )
            continue
        meta = tensor_locations[name]
        inits.append(make_ext_init(name, meta["location"], meta["offset"], meta["length"], meta["dtype"], meta["dims"]))
    if "dflash2.conv_mask_f32" not in tensor_locations:
        inits.append(
            numpy_helper.from_array(
                np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32), name="dflash2.conv_mask_f32"
            )
        )

    inits.extend(
        [
            make_bf16_tensor("dflash2.zero_row_1x2560", np.zeros((1, 2560), dtype=np.uint16), [1, 2560]),
            numpy_helper.from_array(
                np.array([-1, block_size, 4, 160, 1], dtype=np.int64), name="dflash2.shape_m1_4_160_1"
            ),
            numpy_helper.from_array(np.array([-1, 4, 2560], dtype=np.int64), name="dflash2.shape_m1_4_2560"),
            numpy_helper.from_array(np.array([-1, 2560], dtype=np.int64), name="dflash2.shape_m1_2560"),
            numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), name="dflash2.split_4x1"),
        ]
    )
    for lid in range(5):
        for kind in ("attention_conv", "mlp_conv"):
            bname = f"/dflash2/layers.{lid}/{kind}/base_kernel_all_1x4x160x16"
            inits.append(make_bf16_tensor(bname, base_kernels_u16[bname], [1, 1, 4, 160, 16]))
    inits.append(make_bf16_tensor("dflash2.conv_tap_mask_1x8x4x1x1", mask_u16, [1, block_size, 4, 1, 1]))

    graph = helper.make_graph(nodes, "dflash2_graph", inputs, outputs, initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="onnxruntime-genai",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    return model


def build_target_model_proto(
    tensor_locations: dict[str, dict],
    embedded_weights: dict[str, TensorProto],
    kv_scales: dict[str, np.ndarray] | None = None,
    *,
    kv_capacity: int | str = "kv_capacity",
    state_capacity: int | str = "state_capacity",
    max_total_sequence_length: int = 262152,
    indexer_state_capacity_attr: int = 8259,
    verify_rows: int = 8,
    state_update_capacity: int = 7,
) -> onnx.ModelProto:
    """Procedurally construct the unified 958-node Qwen3.8-Flash-Next RC5 `model.onnx` target graph.

    By default, cache capacity dimensions (`kv_capacity` and `state_capacity`) are symbolic and
    `contrib_const_dsa_total_seq` is set to `max_total_sequence_length` (`262152`), allowing a single
    unified `model.onnx` to serve 32K (`num_blocks=142`), 128K (`num_blocks=525`), and 256K (`num_blocks=1037`).
    """
    if kv_scales is None:
        kv_scales = load_calibrated_kv_scales()
    eff_indexer_state_cap = int(state_capacity) if isinstance(state_capacity, int) else int(indexer_state_capacity_attr)
    eff_dsa_total_seq = int(kv_capacity) if isinstance(kv_capacity, int) else int(max_total_sequence_length)

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Add",
            ["committed_length", "sequence_length_minus_one"],
            ["capture.seqlens_k"],
            name="capture/Add_seqlens_k",
        ),
        helper.make_node(
            "Concat",
            ["inputs_embeds", "inputs_embeds", "inputs_embeds", "inputs_embeds"],
            ["v_model.Concat_8"],
            name="model/Concat_node_8",
            axis=-1,
        ),
        helper.make_node("Identity", ["past_position_ids"], ["present_position_ids"], name="node_Identity_16"),
    ]

    base = 87
    prev_streams = "v_model.Concat_8"
    prev_mlp_add = ""
    prev_mlp_mul = ""
    prev_post_mix_id = 0
    dsa_idx = 4

    for i in range(48):
        if i == 1:
            base += 149
        elif i > 1:
            base += 147 if ((i - 1) % 4 == 3) else 109

        sg_base = 0 if i == 0 else (5 + 4 * (i - 1))
        mp_base = 0 if i == 0 else (3 + 2 * (i - 1))

        if i == 1:
            nodes.extend(
                [
                    helper.make_node(
                        "Concat",
                        ["past_key_values.1.ple_context", "input_ids"],
                        ["v_model.layers.1.ple.ple_embedding.Concat_195"],
                        name="model/layers.1/ple/ple_embedding/Concat_node_195",
                        axis=1,
                    ),
                    helper.make_node(
                        "Slice",
                        ["v_model.layers.1.ple.ple_embedding.Concat_195", "const_1d_0", "const_1d_1", "const_1d_3"],
                        ["present.1.ple_context"],
                        name="model/layers.1/ple/ple_embedding/Slice_node_196",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        ["ple_embeddings.1", "merged_proj.2.weight_t.qweight", "merged_proj.2.weight_t.scales"],
                        ["skinny_gemm/4.0/merged_out"],
                        name="skinny_gemm/4.0/MatMulNBits",
                        domain="com.microsoft",
                        K=2560,
                        N=12800,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                    helper.make_node(
                        "Split",
                        ["skinny_gemm/4.0/merged_out", "skinny_gemm/4.0/split_lengths"],
                        ["v_model.layers.1.ple.key_proj.MatMul_198", "v_model.layers.1.ple.value_proj.MatMul_208"],
                        name="skinny_gemm/4.0/Split",
                        axis=-1,
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["v_model.layers.1.ple.key_proj.MatMul_198", "v_model.layers.1.ple.norm_key.Add_200.folded"],
                        ["v_model.layers.1.ple.norm_key.BranchwiseRMSNorm_201"],
                        name="model/layers.1/ple/norm_key/BranchwiseRMSNorm_node_201",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        num_branches=4,
                    ),
                    helper.make_node(
                        "Reshape",
                        ["v_model.layers.1.ple.norm_key.BranchwiseRMSNorm_201", "const_1d_19"],
                        ["v_model.layers.1.ple.Reshape_202"],
                        name="model/layers.1/ple/Reshape_node_202",
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        [
                            "v_model.layers.0.HyperConnectionPostMix_194",
                            "v_model.layers.1.ple.norm_query.Add_204.folded",
                        ],
                        ["v_model.layers.1.ple.norm_query.BranchwiseRMSNorm_205"],
                        name="model/layers.1/ple/norm_query/BranchwiseRMSNorm_node_205",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        num_branches=4,
                    ),
                    helper.make_node(
                        "Reshape",
                        ["v_model.layers.1.ple.norm_query.BranchwiseRMSNorm_205", "const_1d_19"],
                        ["v_model.layers.1.ple.Reshape_206"],
                        name="model/layers.1/ple/Reshape_node_206",
                    ),
                    helper.make_node(
                        "Mul",
                        ["v_model.layers.1.ple.Reshape_202", "v_model.layers.1.ple.Reshape_206"],
                        ["v_model.layers.1.ple.Mul_209"],
                        name="model/layers.1/ple/Mul_node_209",
                    ),
                    helper.make_node(
                        "ReduceSum",
                        ["v_model.layers.1.ple.Mul_209", "const_1d_11"],
                        ["v_model.layers.1.ple.ReduceSum_210"],
                        name="model/layers.1/ple/ReduceSum_node_210",
                        keepdims=1,
                    ),
                    helper.make_node(
                        "CastLike",
                        ["const_50.59644256269407_f32", "v_model.layers.1.ple.ReduceSum_210"],
                        ["v_model.layers.1.ple.CastLike_211"],
                        name="model/layers.1/ple/CastLike_node_211",
                    ),
                    helper.make_node(
                        "Div",
                        ["v_model.layers.1.ple.ReduceSum_210", "v_model.layers.1.ple.CastLike_211"],
                        ["v_model.layers.1.ple.Div_212"],
                        name="model/layers.1/ple/Div_node_212",
                    ),
                    helper.make_node(
                        "Sign",
                        ["v_model.layers.1.ple.Div_212"],
                        ["v_model.layers.1.ple.Sign_213"],
                        name="model/layers.1/ple/Sign_node_213",
                    ),
                    helper.make_node(
                        "Abs",
                        ["v_model.layers.1.ple.Div_212"],
                        ["v_model.layers.1.ple.Abs_214"],
                        name="model/layers.1/ple/Abs_node_214",
                    ),
                    helper.make_node(
                        "CastLike",
                        ["const_1e-06_f32", "v_model.layers.1.ple.Div_212"],
                        ["v_model.layers.1.ple.CastLike_215"],
                        name="model/layers.1/ple/CastLike_node_215",
                    ),
                    helper.make_node(
                        "Max",
                        ["v_model.layers.1.ple.Abs_214", "v_model.layers.1.ple.CastLike_215"],
                        ["v_model.layers.1.ple.Max_216"],
                        name="model/layers.1/ple/Max_node_216",
                    ),
                    helper.make_node(
                        "Sqrt",
                        ["v_model.layers.1.ple.Max_216"],
                        ["v_model.layers.1.ple.Sqrt_217"],
                        name="model/layers.1/ple/Sqrt_node_217",
                    ),
                    helper.make_node(
                        "Mul",
                        ["v_model.layers.1.ple.Sign_213", "v_model.layers.1.ple.Sqrt_217"],
                        ["v_model.layers.1.ple.Mul_218"],
                        name="model/layers.1/ple/Mul_node_218",
                    ),
                    helper.make_node(
                        "Sigmoid",
                        ["v_model.layers.1.ple.Mul_218"],
                        ["v_model.layers.1.ple.Sigmoid_219"],
                        name="model/layers.1/ple/Sigmoid_node_219",
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        ["v_model.layers.1.ple.value_proj.MatMul_208", "const_1d_0"],
                        ["v_model.layers.1.ple.Unsqueeze_220"],
                        name="model/layers.1/ple/Unsqueeze_node_220",
                    ),
                    helper.make_node(
                        "Mul",
                        ["v_model.layers.1.ple.Sigmoid_219", "v_model.layers.1.ple.Unsqueeze_220"],
                        ["v_model.layers.1.ple.Mul_221"],
                        name="model/layers.1/ple/Mul_node_221",
                    ),
                    helper.make_node(
                        "Reshape",
                        ["v_model.layers.1.ple.Mul_221", "const_1d_21"],
                        ["v_model.layers.1.ple.Reshape_222"],
                        name="model/layers.1/ple/Reshape_node_222",
                    ),
                    helper.make_node(
                        "BranchwiseRMSNorm",
                        ["v_model.layers.1.ple.Reshape_222", "v_model.layers.1.ple.norm_conv.Add_224.folded"],
                        ["v_model.layers.1.ple.norm_conv.BranchwiseRMSNorm_225"],
                        name="model/layers.1/ple/norm_conv/BranchwiseRMSNorm_node_225",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        num_branches=4,
                    ),
                    helper.make_node(
                        "CausalConvWithState",
                        [
                            "v_model.layers.1.ple.norm_conv.BranchwiseRMSNorm_225",
                            "model.layers.1.ple.conv1d.weight",
                            "",
                            "past_key_values.1.ple_conv_state",
                        ],
                        [
                            "v_model.layers.1.ple.conv1d.CausalConvWithState_231_0",
                            "present.1.ple_conv_state",
                            "state_update.1.ple_conv_value",
                        ],
                        name="model/layers.1/ple/conv1d/CausalConvWithState_node_231",
                        domain="com.microsoft",
                        activation="silu",
                        channels_last=1,
                        dilation=3,
                        input_channel_offset=0,
                        input_channels=10240,
                        ndim=1,
                        state_channels_last=0,
                        state_length=9,
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "Add",
                        ["v_model.layers.1.ple.Reshape_222", "v_model.layers.1.ple.conv1d.CausalConvWithState_231_0"],
                        ["v_model.layers.1.ple.Add_233"],
                        name="model/layers.1/ple/Add_node_233",
                    ),
                    helper.make_node(
                        "Add",
                        ["v_model.layers.0.HyperConnectionPostMix_194", "v_model.layers.1.ple.Add_233"],
                        ["v_model.layers.1.Add_234"],
                        name="model/layers.1/Add_node_234",
                    ),
                ]
            )
            prev_streams = "v_model.layers.1.Add_234"

        attn_pre = f"v_model.layers.{i}.attn_hyper_connection.HyperConnectionPreMix_{base + 8}"
        attn_mul = f"v_model.layers.{i}.attn_hyper_connection.Mul_{base + 15}"
        hc_in = [
            prev_streams,
            f"v_model.layers.{i}.attn_hyper_connection.hc_norm.Add_{base}.folded",
            f"model.layers.{i}.attn_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
            f"model.layers.{i}.attn_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
            f"model.layers.{i}.attn_hyper_connection.block_inject_weight.weight_t",
        ]
        if i in (0, 1):
            nodes.append(
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    hc_in,
                    [attn_pre, attn_mul],
                    name=f"model/layers.{i}/attn_hyper_connection/HcBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    epsilon=1e-6,
                    gate_div=4.0,
                    gate_mul=2.0,
                    num_branches=4,
                    reduction_scale=0.25,
                    silu_alpha=0.25,
                )
            )
            cur_streams = prev_streams
        else:
            cur_streams = f"v_model.layers.{i - 1}.HyperConnectionPostMix_{prev_post_mix_id}"
            nodes.append(
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    [*hc_in, prev_mlp_add, prev_mlp_mul],
                    [attn_pre, attn_mul, cur_streams],
                    name=f"model/layers.{i}/attn_hyper_connection/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    epsilon=1e-6,
                    gate_div=4.0,
                    gate_mul=2.0,
                    num_branches=4,
                    reduction_scale=0.25,
                    silu_alpha=0.25,
                )
            )

        if i % 4 != 3:
            z_out = f"v_model.layers.{i}.linear_attn.in_proj_z.MatMul_{base + 22}"
            gdn_n = base + 52
            out_proj_out = f"v_model.layers.{i}.linear_attn.out_proj.MatMul_{base + 62}"
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [attn_pre, f"merged_proj.{mp_base}.weight_t.qweight", f"merged_proj.{mp_base}.weight_t.scales"],
                        [f"merged_proj.{mp_base}.out"],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=2560,
                        N=16480,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                    helper.make_node(
                        "Split",
                        [f"merged_proj.{mp_base}.out", "contrib_const_split_gdn"],
                        [
                            f"merged_proj.{mp_base}.out.qkv_10240",
                            z_out,
                            f"merged_proj.{mp_base}.out.a_48",
                            f"merged_proj.{mp_base}.out.b_48",
                        ],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits/SplitGdn",
                        axis=-1,
                    ),
                    helper.make_node(
                        "CausalConvWithState",
                        [
                            f"merged_proj.{mp_base}.out.qkv_10240",
                            f"model.layers.{i}.linear_attn.conv1d.weight",
                            "",
                            f"past_key_values.{i}.conv_state",
                        ],
                        [
                            f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}/GdnProjPrep/conv_qkv_out",
                            f"present.{i}.conv_state",
                            f"state_update.{i}.conv_value",
                        ],
                        name=f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}/GdnProjPrep/CausalConvWithState",
                        domain="com.microsoft",
                        activation="silu",
                        channels_last=1,
                        dilation=1,
                        input_channel_offset=0,
                        input_channels=10240,
                        ndim=1,
                        state_channels_last=0,
                        state_length=4,
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "GatedDeltaNet",
                        [
                            f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}/GdnProjPrep/conv_qkv_out",
                            "",
                            "",
                            "",
                            f"merged_proj.{mp_base}.out.b_48",
                            f"merged_proj.{mp_base}.out.a_48",
                            f"past_key_values.{i}.recurrent_state",
                            f"model.layers.{i}.linear_attn.A_log.f32",
                            f"model.layers.{i}.linear_attn.dt_bias.f32",
                            "state_update_capture_count",
                        ],
                        [
                            f"v_model.layers.{i}.linear_attn.GatedDeltaNet_{gdn_n}_0",
                            f"present.{i}.recurrent_state",
                            f"state_update.{i}.recurrent_capsule",
                        ],
                        name=f"model/layers.{i}/linear_attn/GatedDeltaNet_node_{gdn_n}",
                        domain="com.microsoft",
                        beta_activation="sigmoid",
                        gate_activation="softplus",
                        qk_l2_norm=1,
                        scale=0.0,
                        update_rule="gated_delta",
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"v_model.layers.{i}.linear_attn.GatedDeltaNet_{gdn_n}_0", "const_1d_17"],
                        [f"v_model.layers.{i}.linear_attn.Reshape_{base + 54}"],
                        name=f"model/layers.{i}/linear_attn/Reshape_node_{base + 54}",
                    ),
                    helper.make_node(
                        "Reshape",
                        [z_out, "const_1d_17"],
                        [f"v_model.layers.{i}.linear_attn.Reshape_{base + 55}"],
                        name=f"model/layers.{i}/linear_attn/Reshape_node_{base + 55}",
                    ),
                    helper.make_node(
                        "GatedRMSNorm",
                        [
                            f"v_model.layers.{i}.linear_attn.Reshape_{base + 54}",
                            f"model.layers.{i}.linear_attn.norm.weight",
                            f"v_model.layers.{i}.linear_attn.Reshape_{base + 55}",
                        ],
                        [f"v_model.layers.{i}.linear_attn.norm.GatedRMSNorm_{base + 56}"],
                        name=f"model/layers.{i}/linear_attn/norm/GatedRMSNorm_node_{base + 56}",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        activation="sigmoid",
                    ),
                    helper.make_node(
                        "Shape",
                        [attn_pre],
                        [f"v_model.layers.{i}.linear_attn.Shape_{base + 57}"],
                        name=f"model/layers.{i}/linear_attn/Shape_node_{base + 57}",
                        end=2,
                        start=0,
                    ),
                    helper.make_node(
                        "Concat",
                        [
                            f"v_model.layers.{i}.linear_attn.Shape_{base + 57}",
                            "v_model.layers.0.linear_attn.CastLike_145",
                        ],
                        [f"v_model.layers.{i}.linear_attn.Concat_{base + 59}"],
                        name=f"model/layers.{i}/linear_attn/Concat_node_{base + 59}",
                        axis=0,
                    ),
                    helper.make_node(
                        "Reshape",
                        [
                            f"v_model.layers.{i}.linear_attn.norm.GatedRMSNorm_{base + 56}",
                            f"v_model.layers.{i}.linear_attn.Concat_{base + 59}",
                        ],
                        [f"v_model.layers.{i}.linear_attn.Reshape_{base + 60}"],
                        name=f"model/layers.{i}/linear_attn/Reshape_node_{base + 60}",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        [
                            f"v_model.layers.{i}.linear_attn.Reshape_{base + 60}",
                            f"model.layers.{i}.linear_attn.out_proj.weight_t.qweight",
                            f"model.layers.{i}.linear_attn.out_proj.weight_t.scales",
                        ],
                        [out_proj_out],
                        name=f"skinny_gemm/{sg_base + 1}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=6144,
                        N=2560,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                ]
            )
            mlp_off = 65
        else:
            iqk_out = f"v_model.layers.{i}.self_attn.indexer.index_qk_proj.MatMul_{base + 22}"
            v_out = f"v_model.layers.{i}.self_attn.v_proj.MatMul_{base + 76}"
            out_proj_out = f"v_model.layers.{i}.self_attn.o_proj.MatMul_{base + 100}"
            val_idx = 21 + dsa_idx
            nodes.extend(
                [
                    helper.make_node(
                        "MatMulNBits",
                        [attn_pre, f"merged_proj.{mp_base}.weight_t.qweight", f"merged_proj.{mp_base}.weight_t.scales"],
                        [f"merged_proj.{mp_base}.out"],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=2560,
                        N=13952,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                    helper.make_node(
                        "Split",
                        [f"merged_proj.{mp_base}.out", "contrib_const_split_qsa"],
                        [iqk_out, f"merged_proj.{mp_base}.out.q_gate_12288", f"merged_proj.{mp_base}.out.k_512", v_out],
                        name=f"skinny_gemm/{sg_base}.0/MatMulNBits/SplitQsa",
                        axis=-1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"merged_proj.{mp_base}.out.q_gate_12288", "contrib_const_q_4d"],
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/ReshapeQ4D",
                    ),
                    helper.make_node(
                        "Split",
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d", "const_split_256_256"],
                        [
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/q_only_4d",
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/gate_only_4d",
                        ],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/ReshapeQ4D/SplitQGate",
                        axis=-1,
                    ),
                    helper.make_node(
                        "RMSNormRotary",
                        [
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/q_only_4d",
                            f"v_model.layers.{i}.self_attn.q_norm.Add_{base + 47}.folded",
                            "position_ids",
                            "model.rotary_emb.cos_cache",
                            "model.rotary_emb.sin_cache",
                        ],
                        [f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 73}"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/RMSNormRotaryQ",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        flatten_batch=0,
                        input_start=0,
                        input_width=256,
                        rotary_embedding_dim=64,
                        use_fp32_arithmetic=1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"merged_proj.{mp_base}.out.k_512", "contrib_const_k_4d"],
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/k_4d"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/ReshapeK4D",
                    ),
                    helper.make_node(
                        "RMSNormRotary",
                        [
                            f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/k_4d",
                            f"v_model.layers.{i}.self_attn.k_norm.Add_{base + 61}.folded",
                            "position_ids",
                            "model.rotary_emb.cos_cache",
                            "model.rotary_emb.sin_cache",
                        ],
                        [f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 74}"],
                        name=f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/RMSNormRotaryK",
                        domain="com.microsoft",
                        epsilon=1e-6,
                        flatten_batch=0,
                        input_start=0,
                        input_width=256,
                        rotary_embedding_dim=64,
                        use_fp32_arithmetic=1,
                    ),
                    helper.make_node(
                        "Reshape",
                        [iqk_out, "capture.indexer_qk_shape"],
                        [f"capture.{i}.indexer_qk_flat"],
                        name=f"capture/{i}/Reshape_indexer_qk",
                    ),
                    helper.make_node(
                        "PackedSparseAttentionIndexer",
                        [
                            f"capture.{i}.indexer_qk_flat",
                            "",
                            f"v_model.layers.{i}.self_attn.indexer.q_layernorm.Add_{base + 25}.folded.bf16",
                            f"v_model.layers.{i}.self_attn.indexer.k_layernorm.Add_{base + 29}.folded.bf16",
                            "capture.indexer_cos_cache.folded",
                            "capture.indexer_sin_cache.folded",
                            "cumulative_sequence_lengths",
                            "committed_length",
                            "",
                            "",
                            "",
                            "",
                            f"past_key_values.{i}.index_key",
                            f"past_key_values.{i}.index_buffer",
                            "",
                            f"past_key_values.{i}.index_lengths",
                            "state_update_capture_count",
                        ],
                        [
                            f"v_model.layers.{i}.self_attn.indexer.SparseAttentionIndexer_{base + 33}_0",
                            f"capture.{i}.selected_counts",
                            f"present.{i}.index_key",
                            f"present.{i}.index_buffer",
                            "",
                            f"present.{i}.index_lengths",
                            f"state_update.{i}.indexer_value",
                            "",
                            f"validation.qsa.{i}",
                        ],
                        name=f"model/layers.{i}/self_attn/indexer/PackedQSAIndexer_node_{base + 33}",
                        domain="com.microsoft",
                        compress_ratio=4,
                        epsilon=1e-6,
                        policy_mode="qsa",
                        scale=0.0883883461356163,
                        state_capacity=eff_indexer_state_cap,
                        token_budget=2048,
                        state_update_capacity=state_update_capacity,
                    ),
                    helper.make_node(
                        "DynamicSparseAttention",
                        [
                            f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 73}",
                            f"v_model.layers.{i}.self_attn.RotaryEmbedding_{base + 74}",
                            v_out,
                            f"past_key_values.{i}.key",
                            f"past_key_values.{i}.value",
                            "",
                            "",
                            f"v_model.layers.{i}.self_attn.indexer.SparseAttentionIndexer_{base + 33}_0",
                            f"capture.{i}.selected_counts",
                            "capture.seqlens_k",
                            "contrib_const_dsa_total_seq",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "committed_length",
                            f"node_GuardedDSA_{dsa_idx}_k_scale",
                            f"node_GuardedDSA_{dsa_idx}_v_scale",
                        ],
                        [
                            f"v_model.layers.{i}.self_attn.DynamicSparseAttention_{base + 95}_0",
                            f"present.{i}.key",
                            f"present.{i}.value",
                            f"validation.{val_idx}.present.{i}.key",
                        ],
                        name=f"node_GuardedDSA_{dsa_idx}",
                        domain="com.microsoft",
                        kv_num_heads=2,
                        num_heads=24,
                        scale=0.0625,
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"node_GuardedDSA_{dsa_idx}/QsaProjPrep/q_4d/gate_only_4d", "const_attn_6144_shape"],
                        [f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_3d"],
                        name=f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/ReshapeGate3D",
                    ),
                    helper.make_node(
                        "Sigmoid",
                        [f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_3d"],
                        [f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_sig"],
                        name=f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/SigmoidGate",
                    ),
                    helper.make_node(
                        "Mul",
                        [
                            f"v_model.layers.{i}.self_attn.DynamicSparseAttention_{base + 95}_0",
                            f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/gate_sig",
                        ],
                        [f"v_model.layers.{i}.self_attn.Mul_{base + 98}"],
                        name=f"node_GuardedDSA_{dsa_idx}/SigmoidGateMul/MulGate",
                    ),
                    helper.make_node(
                        "MatMulNBits",
                        [
                            f"v_model.layers.{i}.self_attn.Mul_{base + 98}",
                            f"model.layers.{i}.self_attn.o_proj.weight_t.qweight",
                            f"model.layers.{i}.self_attn.o_proj.weight_t.scales",
                        ],
                        [out_proj_out],
                        name=f"skinny_gemm/{sg_base + 1}.0/MatMulNBits",
                        domain="com.microsoft",
                        K=6144,
                        N=2560,
                        accuracy_level=0,
                        bits=8,
                        block_size=32,
                        weight_prepacked=1,
                    ),
                ]
            )
            dsa_idx += 1
            mlp_off = 103

        mb = base + mlp_off
        mlp_pre = f"v_model.layers.{i}.mlp_hyper_connection.HyperConnectionPreMix_{mb + 8}"
        mlp_mul = f"v_model.layers.{i}.mlp_hyper_connection.Mul_{mb + 15}"
        attn_post = f"v_model.layers.{i}.HyperConnectionPostMix_{mb - 2}"
        mlp_add = f"v_model.layers.{i}.mlp.Add_{mb + 41}"
        nodes.extend(
            [
                helper.make_node(
                    "HyperConnectionLowRankMix",
                    [
                        cur_streams,
                        f"v_model.layers.{i}.mlp_hyper_connection.hc_norm.Add_{mb}.folded",
                        f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                        f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                        f"model.layers.{i}.mlp_hyper_connection.block_inject_weight.weight_t",
                        out_proj_out,
                        attn_mul,
                    ],
                    [mlp_pre, mlp_mul, attn_post],
                    name=f"model/layers.{i}/mlp_hyper_connection/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                    domain="com.microsoft",
                    epsilon=1e-6,
                    gate_div=4.0,
                    gate_mul=2.0,
                    num_branches=4,
                    reduction_scale=0.25,
                    silu_alpha=0.25,
                ),
                helper.make_node(
                    "RouterGateUpProj",
                    [mlp_pre, f"merged_proj.{mp_base + 1}.weight_t"],
                    [
                        f"skinny_gemm/{sg_base + 2}.0/SplitRouterSwishMul/router_3d",
                        f"skinny_gemm/{sg_base + 2}.0/swish_640",
                    ],
                    name=f"skinny_gemm/{sg_base + 2}.0/RouterGateUpProj",
                    domain="com.microsoft",
                    activation="swiglu",
                    activation_alpha=1.0,
                    activation_beta=0.0,
                    limit=0.0,
                    router_size=512,
                ),
                helper.make_node(
                    "Reshape",
                    [f"skinny_gemm/{sg_base + 2}.0/SplitRouterSwishMul/router_3d", "contrib_const_router_2d_shape"],
                    [f"v_model.layers.{i}.mlp.gate.MatMul_{mb + 17}"],
                    name=f"skinny_gemm/{sg_base + 2}.0/SplitRouterSwishMul/ReshapeRouter2D",
                ),
                helper.make_node(
                    "QMoE",
                    [
                        mlp_pre,
                        f"v_model.layers.{i}.mlp.gate.MatMul_{mb + 17}",
                        f"model.layers.{i}.mlp.fc1_experts_weights",
                        f"model.layers.{i}.mlp.fc1_scales",
                        "",
                        f"model.layers.{i}.mlp.fc2_experts_weights",
                        f"model.layers.{i}.mlp.fc2_scales",
                    ],
                    [f"v_model.layers.{i}.mlp.QMoE_{mb + 28}"],
                    name=f"model/layers.{i}/mlp/QMoE_node_{mb + 28}",
                    domain="com.microsoft",
                    activation_type="swiglu",
                    block_size=32,
                    expert_weight_bits=4,
                    k=10,
                    normalize_routing_weights=1,
                    swiglu_fusion=1,
                    weights_prepacked=1,
                ),
                helper.make_node(
                    "MatMul",
                    [
                        f"skinny_gemm/{sg_base + 2}.0/swish_640",
                        f"model.layers.{i}.mlp.shared_expert.down_proj.weight_t",
                    ],
                    [f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/shared_down_out"],
                    name=f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/MatMulBf16",
                ),
                helper.make_node(
                    "GatedAdd",
                    [
                        f"v_model.layers.{i}.mlp.QMoE_{mb + 28}",
                        f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/shared_down_out",
                        mlp_pre,
                        f"model.layers.{i}.mlp.shared_expert_gate.weight_t",
                    ],
                    [mlp_add],
                    name=f"model/layers.{i}/mlp/Add_node_{mb + 41}/fused/shared_down/SharedGateMulAdd/ProjectedSigmoidGateMul/GatedAdd",
                    domain="com.microsoft",
                ),
            ]
        )

        if i == 0:
            pfx0 = "model/layers.0/HyperConnectionPostMix_node_194"
            nodes.extend(
                [
                    helper.make_node(
                        "Reshape",
                        [attn_post, "const_hc_4d_shape"],
                        [f"{pfx0}/streams_4d"],
                        name=f"{pfx0}/ReshapeStreams4D",
                    ),
                    helper.make_node(
                        "Unsqueeze", [mlp_mul, "const_axis_neg1"], [f"{pfx0}/gate_4d"], name=f"{pfx0}/UnsqueezeGate"
                    ),
                    helper.make_node(
                        "Unsqueeze", [mlp_add, "const_axis_neg2"], [f"{pfx0}/block_4d"], name=f"{pfx0}/UnsqueezeBlock"
                    ),
                    helper.make_node(
                        "Mul", [f"{pfx0}/gate_4d", f"{pfx0}/block_4d"], [f"{pfx0}/prod_4d"], name=f"{pfx0}/MulGate"
                    ),
                    helper.make_node(
                        "Add", [f"{pfx0}/streams_4d", f"{pfx0}/prod_4d"], [f"{pfx0}/add_4d"], name=f"{pfx0}/AddStreams"
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"{pfx0}/add_4d", "const_hc_3d_shape"],
                        ["v_model.layers.0.HyperConnectionPostMix_194"],
                        name=f"{pfx0}/ReshapeOut3D",
                    ),
                ]
            )
        prev_streams = attn_post
        prev_mlp_add = mlp_add
        prev_mlp_mul = mlp_mul
        prev_post_mix_id = mb + 42

    pfx_f = "model/hyper_connection_mixer/HcPostBlock/HyperConnectionPostMix"
    val_ins = [f"validation.{25 + k}.present.{3 + 4 * k}.key" for k in range(12)] + [
        f"validation.qsa.{3 + 4 * k}" for k in range(12)
    ]
    nodes.extend(
        [
            helper.make_node(
                "Reshape",
                [prev_streams, "const_hc_4d_shape"],
                [f"{pfx_f}/streams_4d"],
                name=f"{pfx_f}/ReshapeStreams4D",
            ),
            helper.make_node(
                "Unsqueeze", [prev_mlp_mul, "const_axis_neg1"], [f"{pfx_f}/gate_4d"], name=f"{pfx_f}/UnsqueezeGate"
            ),
            helper.make_node(
                "Unsqueeze", [prev_mlp_add, "const_axis_neg2"], [f"{pfx_f}/block_4d"], name=f"{pfx_f}/UnsqueezeBlock"
            ),
            helper.make_node(
                "Mul", [f"{pfx_f}/gate_4d", f"{pfx_f}/block_4d"], [f"{pfx_f}/prod_4d"], name=f"{pfx_f}/MulGate"
            ),
            helper.make_node(
                "Add", [f"{pfx_f}/streams_4d", f"{pfx_f}/prod_4d"], [f"{pfx_f}/add_4d"], name=f"{pfx_f}/AddStreams"
            ),
            helper.make_node(
                "Reshape", [f"{pfx_f}/add_4d", "const_hc_3d_shape"], ["hidden_states"], name=f"{pfx_f}/ReshapeOut3D"
            ),
            helper.make_node(
                "Slice",
                ["hidden_states", "dflash_slice_neg4", "const_1d_1", "const_1d_3"],
                ["hidden_states.last2"],
                name="model/hyper_connection_mixer/Slice_last2",
            ),
            helper.make_node(
                "HyperConnectionLowRankMix",
                [
                    "hidden_states.last2",
                    "v_model.hyper_connection_mixer.hc_norm.Add_5815.folded",
                    "model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16",
                    "model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16",
                    "model.layers.0.attn_hyper_connection.block_inject_weight.weight_t",
                ],
                [
                    "v_model.hyper_connection_mixer.HyperConnectionPreMix_5823",
                    "model/hyper_connection_mixer/HcPostBlock/BranchwiseRMSNorm.dummy_gate",
                ],
                name="model/hyper_connection_mixer/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix",
                domain="com.microsoft",
                epsilon=1e-6,
                gate_div=4.0,
                gate_mul=2.0,
                num_branches=4,
                reduction_scale=0.25,
                silu_alpha=0.25,
            ),
            helper.make_node(
                "MatMulNBits",
                [
                    "v_model.hyper_connection_mixer.HyperConnectionPreMix_5823",
                    "lm_head.weight_t.qweight",
                    "lm_head.weight_t.scales",
                ],
                ["logits"],
                name="lm_head/MatMul_node_5826/SkinnyGemm/MatMulNBits",
                domain="com.microsoft",
                K=2560,
                N=248320,
                accuracy_level=0,
                bits=8,
                block_size=32,
                weight_prepacked=1,
            ),
            helper.make_node("Concat", val_ins, ["validation.aggregate"], name="validation/aggregate", axis=0),
            helper.make_node(
                "ArgMax", ["logits"], ["next_tokens"], name="argmax_rows", axis=-1, keepdims=0, select_last_index=0
            ),
            helper.make_node(
                "SpeculativePrefixCommit",
                [
                    "next_tokens",
                    "input_ids",
                    "validation.aggregate",
                    "fold.ctl",
                    "fold.ctx_commit",
                    "fold.ctx_a",
                    "fold.ptrs",
                ],
                ["fold.tail_done", "fold.ctl_out", "fold.ctx_commit_out"],
                name="fold/SpeculativePrefixCommit",
                domain="com.microsoft",
                head_rows=verify_rows,
            ),
        ]
    )
    for tap_lid in (5, 15, 25, 35, 45):
        post_t = next(
            n.output[2]
            for n in nodes
            if n.name
            == f"model/layers.{tap_lid + 1}/attn_hyper_connection/HcPostBlock/BranchwiseRMSNorm/HyperConnectionLowRankMix"
        )
        nodes.extend(
            [
                helper.make_node(
                    "Reshape",
                    [post_t, "dflash_aux_reshape_4d"],
                    [f"dflash_aux_layer_{tap_lid}_4d"],
                    name=f"dflash_aux/layer_{tap_lid}/Reshape",
                ),
                helper.make_node(
                    "ReduceMean",
                    [f"dflash_aux_layer_{tap_lid}_4d", "dflash_aux_reduce_axis"],
                    [f"dflash_aux_layer_{tap_lid}_mean"],
                    name=f"dflash_aux/layer_{tap_lid}/ReduceMean",
                    keepdims=0,
                ),
            ]
        )
    nodes.append(
        helper.make_node(
            "Concat",
            [f"dflash_aux_layer_{lid}_mean" for lid in (5, 15, 25, 35, 45)],
            ["aux_hidden_states"],
            name="dflash_aux/Concat",
            axis=-1,
        )
    )

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence_length"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence_length"]),
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence_length"]),
        helper.make_tensor_value_info("past_position_ids", TensorProto.INT64, ["batch", kv_capacity]),
    ]
    for i in range(48):
        if i % 4 != 3:
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{i}.conv_state", TensorProto.BFLOAT16, ["batch", 10240, 4]
                )
            )
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{i}.recurrent_state", TensorProto.FLOAT, ["batch", 48, 128, 128]
                )
            )
            if i == 1:
                inputs.extend(
                    [
                        helper.make_tensor_value_info(
                            "ple_embeddings.1", TensorProto.BFLOAT16, ["batch", "sequence_length", 2560]
                        ),
                        helper.make_tensor_value_info(
                            "past_key_values.1.ple_conv_state", TensorProto.BFLOAT16, ["batch", 10240, 9]
                        ),
                        helper.make_tensor_value_info("past_key_values.1.ple_context", TensorProto.INT64, ["batch", 2]),
                    ]
                )
        else:
            inputs.extend(
                [
                    helper.make_tensor_value_info(
                        f"past_key_values.{i}.key", TensorProto.INT8, ["batch", 2, kv_capacity, 256]
                    ),
                    helper.make_tensor_value_info(
                        f"past_key_values.{i}.value", TensorProto.INT8, ["batch", 2, kv_capacity, 256]
                    ),
                    helper.make_tensor_value_info(
                        f"past_key_values.{i}.index_key", TensorProto.FLOAT, ["batch", state_capacity, 128]
                    ),
                ]
            )
    inputs.extend(
        [
            helper.make_tensor_value_info("committed_length", TensorProto.INT32, ["batch"]),
            helper.make_tensor_value_info("sequence_length_minus_one", TensorProto.INT32, ["seq_minus_one_len"]),
            helper.make_tensor_value_info("cumulative_sequence_lengths", TensorProto.INT32, ["batch_plus_one"]),
        ]
    )
    for i in range(3, 48, 4):
        inputs.append(
            helper.make_tensor_value_info(f"past_key_values.{i}.index_buffer", TensorProto.BFLOAT16, ["batch", 7, 128])
        )
        inputs.append(
            helper.make_tensor_value_info(f"past_key_values.{i}.index_lengths", TensorProto.INT32, ["batch", 2])
        )
    inputs.extend(
        [
            helper.make_tensor_value_info("fold.ctl", TensorProto.INT64, ["fold_ctl_len"]),
            helper.make_tensor_value_info("fold.ctx_commit", TensorProto.INT64, ["fold_ctx_commit_len"]),
            helper.make_tensor_value_info("fold.ctx_a", TensorProto.INT64, ["fold_ctx_a_len"]),
            helper.make_tensor_value_info("fold.ptrs", TensorProto.INT64, ["fold_ptrs_len"]),
            helper.make_tensor_value_info(
                "inputs_embeds", TensorProto.BFLOAT16, ["batch_size", "sequence_length", 2560]
            ),
            helper.make_tensor_value_info("state_update_capture_count", TensorProto.INT32, ["batch"]),
        ]
    )

    logits_vi = onnx.ValueInfoProto()
    logits_vi.name = "logits"
    logits_vi.type.tensor_type.elem_type = TensorProto.BFLOAT16
    outputs = [
        logits_vi,
        helper.make_tensor_value_info("present_position_ids", TensorProto.INT64, ["batch", kv_capacity]),
        helper.make_tensor_value_info("hidden_states", TensorProto.BFLOAT16, ["batch", "sequence_length", 10240]),
    ]
    for i in range(48):
        if i % 4 != 3:
            outputs.append(
                helper.make_tensor_value_info(f"present.{i}.conv_state", TensorProto.BFLOAT16, ["batch", 10240, 4])
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"present.{i}.recurrent_state", TensorProto.FLOAT, ["batch", 48, 128, 128]
                )
            )
            if i == 1:
                outputs.append(
                    helper.make_tensor_value_info("present.1.ple_conv_state", TensorProto.BFLOAT16, ["batch", 10240, 9])
                )
                outputs.append(
                    helper.make_tensor_value_info("present.1.ple_context", TensorProto.INT64, ["batch", "_d12"])
                )
        else:
            outputs.extend(
                [
                    helper.make_tensor_value_info(f"present.{i}.key", TensorProto.INT8, ["batch", 2, kv_capacity, 256]),
                    helper.make_tensor_value_info(
                        f"present.{i}.value", TensorProto.INT8, ["batch", 2, kv_capacity, 256]
                    ),
                    helper.make_tensor_value_info(
                        f"present.{i}.index_key", TensorProto.FLOAT, ["batch", state_capacity, 128]
                    ),
                ]
            )
    outputs.append(helper.make_tensor_value_info("validation.aggregate", TensorProto.INT32, ["validation_count"]))
    for i in range(3, 48, 4):
        outputs.append(
            helper.make_tensor_value_info(f"present.{i}.index_buffer", TensorProto.BFLOAT16, ["batch", 7, 128])
        )
        outputs.append(helper.make_tensor_value_info(f"present.{i}.index_lengths", TensorProto.INT32, ["batch", 2]))
    outputs.extend(
        [
            helper.make_tensor_value_info("next_tokens", TensorProto.INT64, ["batch", "sequence_length"]),
            helper.make_tensor_value_info("fold.tail_done", TensorProto.INT32, ["batch"]),
            helper.make_tensor_value_info("fold.ctl_out", TensorProto.INT64, ["fold_ctl_len"]),
            helper.make_tensor_value_info("fold.ctx_commit_out", TensorProto.INT64, ["fold_ctx_commit_len"]),
        ]
    )
    for i in range(48):
        if i == 1:
            outputs.append(
                helper.make_tensor_value_info(
                    "state_update.1.ple_conv_value", TensorProto.BFLOAT16, ["batch", state_update_capacity, 10240]
                )
            )
        if i % 4 != 3:
            outputs.append(
                helper.make_tensor_value_info(
                    f"state_update.{i}.conv_value", TensorProto.BFLOAT16, ["batch", state_update_capacity, 10240]
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"state_update.{i}.recurrent_capsule", TensorProto.FLOAT, ["batch", state_update_capacity * 8240]
                )
            )
        else:
            outputs.append(
                helper.make_tensor_value_info(
                    f"state_update.{i}.indexer_value", TensorProto.BFLOAT16, ["batch", state_update_capacity, 128]
                )
            )
    outputs.append(
        helper.make_tensor_value_info("aux_hidden_states", TensorProto.BFLOAT16, ["batch", "sequence_length", 12800])
    )

    inits: list[TensorProto] = [
        numpy_helper.from_array(np.array([-2], dtype=np.int64), name="const_1d_0"),
        numpy_helper.from_array(np.array([9223372036854775807], dtype=np.int64), name="const_1d_1"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name="const_1d_3"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="const_1d_11"),
        numpy_helper.from_array(np.array(1e-6, dtype=np.float32), name="const_1e-06_f32"),
        numpy_helper.from_array(np.array([-1, 128], dtype=np.int64), name="const_1d_17"),
        numpy_helper.from_array(np.array([0, 0, 4, 2560], dtype=np.int64), name="const_1d_19"),
        numpy_helper.from_array(np.array(50.59644256269407, dtype=np.float32), name="const_50.59644256269407_f32"),
        numpy_helper.from_array(np.array([0, 0, -1], dtype=np.int64), name="const_1d_21"),
        numpy_helper.from_array(np.array([6144], dtype=np.int64), name="v_model.layers.0.linear_attn.CastLike_145"),
        numpy_helper.from_array(np.array([-1, 640], dtype=np.int64), name="capture.indexer_qk_shape"),
        numpy_helper.from_array(np.array([10240, 2560], dtype=np.int64), name="skinny_gemm/4.0/split_lengths"),
        numpy_helper.from_array(np.array([0, 0, 24, 512], dtype=np.int64), name="contrib_const_q_4d"),
        numpy_helper.from_array(np.array([0, 0, 2, 256], dtype=np.int64), name="contrib_const_k_4d"),
        numpy_helper.from_array(np.array([eff_dsa_total_seq], dtype=np.int32), name="contrib_const_dsa_total_seq"),
        numpy_helper.from_array(np.array([10240, 6144, 48, 48], dtype=np.int64), name="contrib_const_split_gdn"),
        numpy_helper.from_array(np.array([640, 12288, 512, 512], dtype=np.int64), name="contrib_const_split_qsa"),
        numpy_helper.from_array(np.array([-1, 512], dtype=np.int64), name="contrib_const_router_2d_shape"),
        numpy_helper.from_array(np.array([-verify_rows], dtype=np.int64), name="dflash_slice_neg4"),
        numpy_helper.from_array(np.array([0, 0, 4, 2560], dtype=np.int64), name="dflash_aux_reshape_4d"),
        numpy_helper.from_array(np.array([-2], dtype=np.int64), name="dflash_aux_reduce_axis"),
        numpy_helper.from_array(np.array([0, 0, 4, 2560], dtype=np.int64), name="const_hc_4d_shape"),
        numpy_helper.from_array(np.array([0, 0, 10240], dtype=np.int64), name="const_hc_3d_shape"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name="const_axis_neg1"),
        numpy_helper.from_array(np.array([-2], dtype=np.int64), name="const_axis_neg2"),
        numpy_helper.from_array(np.array([256, 256], dtype=np.int64), name="const_split_256_256"),
        numpy_helper.from_array(np.array([0, 0, 6144], dtype=np.int64), name="const_attn_6144_shape"),
        numpy_helper.from_array(np.array([512, 1280], dtype=np.int64), name="const_split_512_1280"),
    ]

    for k_name, arr in sorted(kv_scales.items()):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=k_name))
    for _k_name, t_proto in sorted(embedded_weights.items()):
        t_copy = TensorProto()
        t_copy.CopyFrom(t_proto)
        inits.append(t_copy)

    used_inputs = {inp for n in nodes for inp in n.input if inp}
    existing_init_names = {t.name for t in inits}
    for name in sorted(used_inputs):
        if name in existing_init_names:
            continue
        if name in tensor_locations:
            meta = tensor_locations[name]
            inits.append(
                make_ext_init(name, meta["location"], meta["offset"], meta["length"], meta["dtype"], meta["dims"])
            )

    graph = helper.make_graph(nodes, "qwen4_exp", inputs, outputs, initializer=inits)
    model = helper.make_model(
        graph,
        producer_name="mobius",
        opset_imports=[helper.make_opsetid("", 24), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11
    fixed_kv_map = {
        **{
            f"past_key_values.{lid}.{sfx}": f"present.{lid}.{sfx}"
            for lid in sorted((3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47), key=str)
            for sfx in ("key", "value")
        },
        "past_position_ids": "present_position_ids",
    }
    meta_props = (
        ("experiment.cuda_graph_device_lengths", "committed_length + sequence_length_minus_one"),
        ("experiment.cuda_graph_index_rotary", "static-expanded-initializer"),
        ("experiment.cuda_graph_index_state_type", "float32"),
        ("experiment.cuda_graph_indexer", "qwen.capture::PackedQSAIndexer;device lengths;compressed state"),
        ("experiment.cuda_graph_ple_empty_branch", "pruned"),
        ("experiment.cuda_graph_sequence_length", "per fixed binding"),
        ("experiment.fixed_kv_capacity", str(eff_dsa_total_seq)),
        ("experiment.fixed_kv_logical_length", "attention_mask.shape[1]"),
        ("experiment.fixed_kv_mapping", json.dumps(fixed_kv_map)),
        ("experiment.host_positions", "1"),
        (
            "experiment.static_execution",
            "fixedKV+index+positions;immutable-recurrent;deferred-status;no-fullcapture",
        ),
        ("experiment.validation_status_names", json.dumps(val_ins)),
        (
            "mobius.cache_abi",
            "qwen4-exp:position_ids;linear=conv_state,recurrent_state[,ple_conv_state,ple_context];qsa=key,value,index_key",
        ),
        (
            "mobius.hidden_states_abi",
            "pre-final-hyper-mixer;unpruned-sequence;row=tokens-consumed;mtp-input=(hidden[t],token[t+1])",
        ),
        ("mobius.logits_abi", "sequence-row-gather:input=logits_indices;axis=1;full-vocabulary"),
        ("mobius.ple_abi", "external-cpu-fp8:scaled-bfloat16;input=ple_embeddings.{zero_based_layer}"),
    )
    for k_prop, v_prop in meta_props:
        prop = model.metadata_props.add()
        prop.key = k_prop
        prop.value = v_prop
    return model


def _load_safetensors_tensor(shard_path: Path, key: str, device: str = "cpu") -> torch.Tensor:
    with open(shard_path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    info = hdr[key]
    s, e = info["data_offsets"]
    shape = tuple(info["shape"])
    dtype_str = info["dtype"]
    base_off = 8 + hlen + s
    if dtype_str == "BF16":
        mm = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=base_off, shape=shape)
        t = torch.from_numpy(mm.copy()).view(torch.bfloat16)
    elif dtype_str == "F32":
        mm = np.memmap(shard_path, dtype=np.float32, mode="r", offset=base_off, shape=shape)
        t = torch.from_numpy(mm.copy())
    elif dtype_str == "I64":
        mm = np.memmap(shard_path, dtype=np.int64, mode="r", offset=base_off, shape=shape)
        t = torch.from_numpy(mm.copy())
    else:
        raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {key}")
    return t.to(device=device) if device != "cpu" else t


class CheckpointTensorLoader:
    """Memory-mapped reader for single-file (`model.safetensors`) or sharded (`model.safetensors.index.json`) checkpoints."""

    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir.resolve()
        idx_path = self.ckpt_dir / "model.safetensors.index.json"
        single_path = self.ckpt_dir / "model.safetensors"
        if idx_path.is_file():
            self.weight_map: dict[str, str] = json.loads(idx_path.read_text())["weight_map"]
        elif single_path.is_file():
            with open(single_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            self.weight_map = {k: "model.safetensors" for k in hdr if k != "__metadata__"}
        else:
            raise FileNotFoundError(
                f"Neither model.safetensors.index.json nor model.safetensors found in {self.ckpt_dir}"
            )
        self._headers: dict[str, tuple[int, dict]] = {}

    def _resolve_key(self, name: str) -> str:
        if name in self.weight_map:
            return name
        if name.startswith("model.layers."):
            alt = "model.language_model." + name[len("model.") :]
            if alt in self.weight_map:
                return alt
        if name.startswith("model.embed_tokens."):
            alt = "model.language_model." + name[len("model.") :]
            if alt in self.weight_map:
                return alt
        if name.startswith("model.hyper_connection_mixer."):
            alt = "model.language_model." + name[len("model.") :]
            if alt in self.weight_map:
                return alt
        raise KeyError(f"Weight {name!r} not found in {self.ckpt_dir}")

    def read_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        key = self._resolve_key(name)
        shard_file = self.weight_map[key]
        if shard_file not in self._headers:
            p = self.ckpt_dir / shard_file
            with open(p, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
            self._headers[shard_file] = (8 + hlen, hdr)
        base_off, hdr = self._headers[shard_file]
        info = hdr[key]
        s, _e = info["data_offsets"]
        shape = tuple(info["shape"])
        dtype_str = info["dtype"]
        p = self.ckpt_dir / shard_file
        if dtype_str == "BF16":
            mm = np.memmap(p, dtype=np.uint16, mode="r", offset=base_off + s, shape=shape)
            t = torch.from_numpy(mm).view(torch.bfloat16)
        elif dtype_str == "F32":
            mm = np.memmap(p, dtype=np.float32, mode="r", offset=base_off + s, shape=shape)
            t = torch.from_numpy(mm)
        elif dtype_str == "I64":
            mm = np.memmap(p, dtype=np.int64, mode="r", offset=base_off + s, shape=shape)
            t = torch.from_numpy(mm)
        else:
            raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {key}")
        return t.to(device=device, non_blocking=False) if device != "cpu" else t.clone()


@torch.no_grad()
def build_embedding_from_hf(
    hf_loader: CheckpointTensorLoader, out_dir: Path, vocab_size: int = 248320, hidden_size: int = 2560
) -> dict[str, dict]:
    """Quantize `model.language_model.embed_tokens.weight` to per-row symmetric INT8 in `embedding.onnx.data` and save `embedding.onnx`."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb_f32 = hf_loader.read_tensor("model.language_model.embed_tokens.weight", device=dev).float()
    sc = (emb_f32.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
    q_i8 = torch.round(emb_f32 / sc).clamp(-127.0, 127.0).to(torch.int8).cpu().numpy()
    sc_np = sc.to(torch.float32).cpu().numpy()

    i8_bytes = vocab_size * hidden_size
    s_bytes = vocab_size * 4
    data_path = out_dir / "embedding.onnx.data"
    with open(data_path, "wb") as f:
        f.write(q_i8.tobytes())
        f.write(sc_np.tobytes())

    emb_proto = build_embedding_model_proto(vocab_size=vocab_size, hidden_size=hidden_size)
    onnx.save_model(emb_proto, str(out_dir / "embedding.onnx"), save_as_external_data=False)
    return {
        "model.embed_tokens.weight": {
            "location": "embedding.onnx.data",
            "offset": 0,
            "length": i8_bytes,
            "dtype": TensorProto.INT8,
            "dims": [vocab_size, hidden_size],
        },
        "model.embed_tokens.weight.row_scales": {
            "location": "embedding.onnx.data",
            "offset": i8_bytes,
            "length": s_bytes,
            "dtype": TensorProto.FLOAT,
            "dims": [vocab_size, 1],
        },
    }


@torch.no_grad()
def build_ple_int8_from_hf(hf_loader: CheckpointTensorLoader, out_dir: Path, eos_token_id: int = 248044) -> None:
    """Quantize all 128 PLE n-gram shards from `hf_loader` to `ple/ple_int8_00..09.data` and write `ple_table.txt`."""
    ple_dir = out_dir / "ple"
    ple_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mults = [
        int(x)
        for x in hf_loader.read_tensor("model.language_model.layers.1.ple.ple_embedding.layer_multipliers").tolist()
    ]
    if len(mults) == 3:
        mults.append(0)
    vocabs = [
        int(x)
        for x in hf_loader.read_tensor(
            "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"
        ).tolist()
    ]
    offsets = [
        int(x)
        for x in hf_loader.read_tensor("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets").tolist()
    ]

    lines = [
        "qwen38_ple_table 1",
        f"n_heads {len(vocabs)}",
        "head_dim 160",
        "heads_per_ngram 8",
        f"eos {eos_token_id}",
        f"scale -{PLE_INT8_SCALE:.12f}",
        "mult " + " ".join(str(x) for x in mults),
        "vocab " + " ".join(str(x) for x in vocabs),
        "offset " + " ".join(str(x) for x in offsets),
    ]

    shards_per_file = [13] * 8 + [12] * 2
    file_idx = 0
    in_file_count = 0
    cur_f = None
    cur_rel = ""
    cur_off = 0
    inv_scale = 1.0 / PLE_INT8_SCALE

    try:
        for s in range(128):
            if cur_f is None:
                cur_rel = f"ple/ple_int8_{file_idx:02d}.data"
                cur_f = open(out_dir / cur_rel, "wb")  # noqa: SIM115
                cur_off = 0
                in_file_count = 0
            t = hf_loader.read_tensor(
                f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight", device=dev
            )
            rows, _width = int(t.shape[0]), int(t.shape[1])
            q_i8 = torch.round(t.float() * inv_scale).clamp(-127.0, 127.0).to(torch.int8).cpu().numpy()
            raw_bytes = q_i8.tobytes()
            cur_f.write(raw_bytes)
            lines.append(f"shard {cur_rel} {cur_off} {rows}")
            cur_off += len(raw_bytes)
            in_file_count += 1
            if in_file_count == shards_per_file[file_idx]:
                cur_f.close()
                cur_f = None
                file_idx += 1
    finally:
        if cur_f is not None:
            cur_f.close()

    (out_dir / "ple_table.txt").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def build_target_non_qmoe_from_hf(
    hf_loader: CheckpointTensorLoader,
    out_dir: Path,
    fast_quant_prepack_fn: Callable,
    max_rotary_rows: int = 262144,
) -> tuple[dict[str, dict], dict[str, TensorProto]]:
    """Build `norm_and_rope.data`, `attn_consts.data`, `contrib_weights.data`, `matmul_int8.data`, `matmul_scales.data`
    and embedded layer norms directly from `hf_loader`.
    """
    tensor_locations: dict[str, dict] = {}
    embedded_weights: dict[str, TensorProto] = {}

    norm_writer = BinaryBlobWriter(out_dir / "norm_and_rope.data")
    attn_writer = BinaryBlobWriter(out_dir / "attn_consts.data")
    contrib_writer = BinaryBlobWriter(out_dir / "contrib_weights.data")
    mm_w_writer = BinaryBlobWriter(out_dir / "matmul_int8.data")
    mm_s_writer = BinaryBlobWriter(out_dir / "matmul_scales.data")

    try:
        # 1. 256K RoPE caches in norm_and_rope.data and attn_consts.data
        dim = 64
        inv = 1.0 / (10000000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(max_rotary_rows, dtype=torch.float32)[:, None]
        freqs = pos * inv[None, :]
        cos32 = torch.cos(freqs).to(torch.bfloat16)
        sin32 = torch.sin(freqs).to(torch.bfloat16)
        cos64 = torch.cat([cos32, cos32], dim=-1).contiguous()
        sin64 = torch.cat([sin32, sin32], dim=-1).contiguous()

        tensor_locations["model.rotary_emb.cos_cache"] = norm_writer.write_aligned(
            cos32.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 32]
        )
        tensor_locations["model.rotary_emb.sin_cache"] = norm_writer.write_aligned(
            sin32.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 32]
        )
        tensor_locations["capture.indexer_cos_cache.folded"] = attn_writer.write_aligned(
            cos64.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 64]
        )
        tensor_locations["capture.indexer_sin_cache.folded"] = attn_writer.write_aligned(
            sin64.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, [max_rotary_rows, 64]
        )

        def write_bf16(writer: BinaryBlobWriter, t: torch.Tensor) -> dict:
            t_c = t.to(torch.bfloat16).contiguous().cpu()
            return writer.write_aligned(t_c.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(t_c.shape))

        def write_int8_matmul(name_base: str, w_nk: torch.Tensor) -> None:
            qb, qs = fast_quant_prepack_fn(w_nk, bits=8, block_size=32, is_qmoe=False)
            qw_name = f"{name_base}.qweight"
            qs_name = f"{name_base}.scales"
            tensor_locations[qw_name] = mm_w_writer.write_aligned(
                qb.view(torch.uint8).numpy().tobytes(), TensorProto.UINT8, list(qb.shape)
            )
            tensor_locations[qs_name] = mm_s_writer.write_aligned(
                qs.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(qs.shape)
            )

        for i in range(48):
            base = layer_base_node_id(i)
            mp_base = 0 if i == 0 else (3 + 2 * (i - 1))
            lpfx = f"model.language_model.layers.{i}"

            if i == 1:
                # PLE norms, conv1d, and merged key_proj + value_proj
                for p_key, oname in (
                    (f"{lpfx}.ple.norm_key.weight", "v_model.layers.1.ple.norm_key.Add_200.folded"),
                    (f"{lpfx}.ple.norm_query.weight", "v_model.layers.1.ple.norm_query.Add_204.folded"),
                    (f"{lpfx}.ple.norm_conv.weight", "v_model.layers.1.ple.norm_conv.Add_224.folded"),
                ):
                    w_n = (1.0 + hf_loader.read_tensor(p_key).float()).to(torch.bfloat16)
                    embedded_weights[oname] = make_bf16_tensor(oname, w_n.view(torch.uint16).numpy(), list(w_n.shape))
                ple_conv = hf_loader.read_tensor(f"{lpfx}.ple.conv1d.weight")
                tensor_locations["model.layers.1.ple.conv1d.weight"] = write_bf16(norm_writer, ple_conv)
                ple_kv = torch.cat(
                    [
                        hf_loader.read_tensor(f"{lpfx}.ple.key_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.ple.value_proj.weight"),
                    ],
                    dim=0,
                )
                write_int8_matmul("merged_proj.2.weight_t", ple_kv)

            # Attn hyper-connection
            w_hc_a = (1.0 + hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.hc_norm.weight").float()).to(
                torch.bfloat16
            )
            hc_a_name = f"v_model.layers.{i}.attn_hyper_connection.hc_norm.Add_{base}.folded"
            embedded_weights[hc_a_name] = make_bf16_tensor(
                hc_a_name, w_hc_a.view(torch.uint16).numpy(), list(w_hc_a.shape)
            )

            tensor_locations[
                f"model.layers.{i}.attn_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.input_mix_weight_down.weight").T
            )
            tensor_locations[
                f"model.layers.{i}.attn_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.input_mix_weight_up.weight").T
            )
            tensor_locations[f"model.layers.{i}.attn_hyper_connection.block_inject_weight.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.attn_hyper_connection.block_inject_weight.weight").T
            )

            if i % 4 != 3:
                w_in = torch.cat(
                    [
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_qkv.weight"),
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_z.weight"),
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_b.weight"),
                        hf_loader.read_tensor(f"{lpfx}.linear_attn.in_proj_a.weight"),
                    ],
                    dim=0,
                )
                write_int8_matmul(f"merged_proj.{mp_base}.weight_t", w_in)
                write_int8_matmul(
                    f"model.layers.{i}.linear_attn.out_proj.weight_t",
                    hf_loader.read_tensor(f"{lpfx}.linear_attn.out_proj.weight"),
                )
                tensor_locations[f"model.layers.{i}.linear_attn.conv1d.weight"] = write_bf16(
                    norm_writer, hf_loader.read_tensor(f"{lpfx}.linear_attn.conv1d.weight")
                )
                tensor_locations[f"model.layers.{i}.linear_attn.norm.weight"] = write_bf16(
                    norm_writer, hf_loader.read_tensor(f"{lpfx}.linear_attn.norm.weight")
                )
                a_log = hf_loader.read_tensor(f"{lpfx}.linear_attn.A_log").float().numpy()
                dt_bias = hf_loader.read_tensor(f"{lpfx}.linear_attn.dt_bias").float().numpy()
                embedded_weights[f"model.layers.{i}.linear_attn.A_log.f32"] = numpy_helper.from_array(
                    a_log, name=f"model.layers.{i}.linear_attn.A_log.f32"
                )
                embedded_weights[f"model.layers.{i}.linear_attn.dt_bias.f32"] = numpy_helper.from_array(
                    dt_bias, name=f"model.layers.{i}.linear_attn.dt_bias.f32"
                )
                mlp_off = 65
            else:
                w_qsa = torch.cat(
                    [
                        hf_loader.read_tensor(f"{lpfx}.self_attn.indexer.index_qk_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.self_attn.q_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.self_attn.k_proj.weight"),
                        hf_loader.read_tensor(f"{lpfx}.self_attn.v_proj.weight"),
                    ],
                    dim=0,
                )
                write_int8_matmul(f"merged_proj.{mp_base}.weight_t", w_qsa)
                write_int8_matmul(
                    f"model.layers.{i}.self_attn.o_proj.weight_t",
                    hf_loader.read_tensor(f"{lpfx}.self_attn.o_proj.weight"),
                )
                iq_n = (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.indexer.q_layernorm.weight").float()).to(
                    torch.bfloat16
                )
                ik_n = (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.indexer.k_layernorm.weight").float()).to(
                    torch.bfloat16
                )
                iq_name = f"v_model.layers.{i}.self_attn.indexer.q_layernorm.Add_{base + 25}.folded.bf16"
                ik_name = f"v_model.layers.{i}.self_attn.indexer.k_layernorm.Add_{base + 29}.folded.bf16"
                embedded_weights[iq_name] = make_bf16_tensor(iq_name, iq_n.view(torch.uint16).numpy(), list(iq_n.shape))
                embedded_weights[ik_name] = make_bf16_tensor(ik_name, ik_n.view(torch.uint16).numpy(), list(ik_n.shape))

                qn_f32 = (
                    (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.q_norm.weight").float()).numpy().astype(np.float32)
                )
                kn_f32 = (
                    (1.0 + hf_loader.read_tensor(f"{lpfx}.self_attn.k_norm.weight").float()).numpy().astype(np.float32)
                )
                qn_name = f"v_model.layers.{i}.self_attn.q_norm.Add_{base + 47}.folded"
                kn_name = f"v_model.layers.{i}.self_attn.k_norm.Add_{base + 61}.folded"
                embedded_weights[qn_name] = numpy_helper.from_array(qn_f32, name=qn_name)
                embedded_weights[kn_name] = numpy_helper.from_array(kn_f32, name=kn_name)
                mlp_off = 103

            mb = base + mlp_off
            w_hc_m = (1.0 + hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.hc_norm.weight").float()).to(
                torch.bfloat16
            )
            hc_m_name = f"v_model.layers.{i}.mlp_hyper_connection.hc_norm.Add_{mb}.folded"
            embedded_weights[hc_m_name] = make_bf16_tensor(
                hc_m_name, w_hc_m.view(torch.uint16).numpy(), list(w_hc_m.shape)
            )

            tensor_locations[
                f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.input_mix_weight_down.weight").T
            )
            tensor_locations[
                f"model.layers.{i}.mlp_hyper_connection.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16"
            ] = write_bf16(
                contrib_writer, hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.input_mix_weight_up.weight").T
            )
            tensor_locations[f"model.layers.{i}.mlp_hyper_connection.block_inject_weight.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.mlp_hyper_connection.block_inject_weight.weight").T
            )

            w_rgu = torch.cat(
                [
                    hf_loader.read_tensor(f"{lpfx}.mlp.gate.weight"),
                    hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert.gate_proj.weight"),
                    hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert.up_proj.weight"),
                ],
                dim=0,
            ).T
            tensor_locations[f"merged_proj.{mp_base + 1}.weight_t"] = write_bf16(norm_writer, w_rgu)
            tensor_locations[f"model.layers.{i}.mlp.shared_expert.down_proj.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert.down_proj.weight").T
            )
            tensor_locations[f"model.layers.{i}.mlp.shared_expert_gate.weight_t"] = write_bf16(
                norm_writer, hf_loader.read_tensor(f"{lpfx}.mlp.shared_expert_gate.weight").T
            )

        # Final hyper-connection mixer & LM head
        w_hc_f = (1.0 + hf_loader.read_tensor("model.language_model.hyper_connection_mixer.hc_norm.weight").float()).to(
            torch.bfloat16
        )
        hc_f_name = "v_model.hyper_connection_mixer.hc_norm.Add_5815.folded"
        embedded_weights[hc_f_name] = make_bf16_tensor(hc_f_name, w_hc_f.view(torch.uint16).numpy(), list(w_hc_f.shape))
        tensor_locations["model.hyper_connection_mixer.input_mix_weight_down.weight_t.hc_tiled_down.untiled_bf16"] = (
            write_bf16(
                contrib_writer,
                hf_loader.read_tensor("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight").T,
            )
        )
        tensor_locations["model.hyper_connection_mixer.input_mix_weight_up.weight_t.hc_tiled_up.untiled_bf16"] = (
            write_bf16(
                contrib_writer,
                hf_loader.read_tensor("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight").T,
            )
        )
        write_int8_matmul("lm_head.weight_t", hf_loader.read_tensor("lm_head.weight"))
    finally:
        norm_writer.close()
        attn_writer.close()
        contrib_writer.close()
        mm_w_writer.close()
        mm_s_writer.close()

    return tensor_locations, embedded_weights


@torch.no_grad()
def build_dflash2_from_hf(
    dflash_dir: Path,
    out_dir: Path,
    shared_locations: dict[str, dict],
    fast_quant_prepack_fn: Callable,
    max_position: int = 262148,
) -> dict[str, dict]:
    """Quantize and pack `dflash2.onnx.data` and save `dflash2.onnx` from `dflash_dir`."""
    dflash_loader = CheckpointTensorLoader(dflash_dir)
    cfg = json.loads((dflash_dir / "config.json").read_text())
    dfl_cfg = cfg.get("dflash_config", {})
    block_size = int(dfl_cfg.get("block_size", 8))
    num_draft_tokens = block_size - 1
    selector_top_k = int(dfl_cfg.get("selector_top_k", 16))
    sliding_window = int(cfg.get("sliding_window", 2048))
    rope_theta = float(cfg.get("rope_parameters", {}).get("rope_theta", 10000.0))
    head_dim = int(cfg.get("head_dim", 128))

    writer = BinaryBlobWriter(out_dir / "dflash2.onnx.data")
    dflash_locations: dict[str, dict] = {
        "model.embed_tokens.weight": shared_locations["model.embed_tokens.weight"],
        "model.embed_tokens.weight.row_scales": shared_locations["model.embed_tokens.weight.row_scales"],
        "lm_head.weight_t.qweight": shared_locations["lm_head.weight_t.qweight"],
        "lm_head.weight_t.scales": shared_locations["lm_head.weight_t.scales"],
    }
    base_kernels_u16: dict[str, np.ndarray] = {}

    try:

        def write_int4(name_base: str, w_nk: torch.Tensor) -> None:
            qb, qs = fast_quant_prepack_fn(w_nk.contiguous(), bits=4, block_size=32, is_qmoe=False)
            dflash_locations[f"{name_base}.qweight"] = writer.write_aligned(
                qb.view(torch.uint8).numpy().tobytes(), TensorProto.UINT8, list(qb.shape)
            )
            dflash_locations[f"{name_base}.scales"] = writer.write_aligned(
                qs.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, list(qs.shape)
            )

        def write_bf16(name: str, t: torch.Tensor, dims: list[int] | None = None) -> None:
            t_c = t.to(torch.bfloat16).contiguous().cpu()
            dflash_locations[name] = writer.write_aligned(
                t_c.view(torch.uint16).numpy().tobytes(), TensorProto.BFLOAT16, dims or list(t_c.shape)
            )

        write_int4("dflash2.fc.MatMul.weight", dflash_loader.read_tensor("fc.weight"))
        for lid in range(5):
            w_qkv = torch.cat(
                [
                    dflash_loader.read_tensor(f"layers.{lid}.self_attn.q_proj.weight"),
                    dflash_loader.read_tensor(f"layers.{lid}.self_attn.k_proj.weight"),
                    dflash_loader.read_tensor(f"layers.{lid}.self_attn.v_proj.weight"),
                ],
                dim=0,
            )
            write_int4(f"dflash2.layers.{lid}.attn.qkv_proj.MatMul.weight", w_qkv)
            write_int4(
                f"dflash2.layers.{lid}.attn.o_proj.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.self_attn.o_proj.weight"),
            )
            w_gu = torch.cat(
                [
                    dflash_loader.read_tensor(f"layers.{lid}.mlp.gate_proj.weight"),
                    dflash_loader.read_tensor(f"layers.{lid}.mlp.up_proj.weight"),
                ],
                dim=0,
            )
            write_int4(f"dflash2.layers.{lid}.mlp.gate_up_proj.MatMul.weight", w_gu)
            write_int4(
                f"dflash2.layers.{lid}.mlp.down_proj.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.mlp.down_proj.weight"),
            )

        for lid in range(5):
            for kind in ("attention_conv", "mlp_conv"):
                bk = dflash_loader.read_tensor(f"layers.{lid}.{kind}.base_kernel").to(torch.bfloat16).contiguous()
                pfx = f"/dflash2/layers.{lid}/{kind}"
                write_bf16(f"{pfx}/finish.base_kernel.0", bk[1, 0].unsqueeze(0), [1, 160, 16])
                write_bf16(f"{pfx}/finish.base_kernel.1", bk[1, 1].unsqueeze(0), [1, 160, 16])
                write_bf16(f"{pfx}/prepare.base_kernel.0", bk[0, 0].unsqueeze(0), [1, 160, 16])
                write_bf16(f"{pfx}/prepare.base_kernel.1", bk[0, 1].unsqueeze(0), [1, 160, 16])
                base_kernels_u16[f"{pfx}/base_kernel_all_1x4x160x16"] = (
                    bk.reshape(1, 1, 4, 160, 16).view(torch.uint16).numpy().copy()
                )

        write_bf16(
            "dflash2.candidate_selector.predecessor_codebook",
            dflash_loader.read_tensor("candidate_selector.predecessor_codebook"),
        )
        write_bf16(
            "dflash2.candidate_selector.successor_codebook",
            dflash_loader.read_tensor("candidate_selector.successor_codebook"),
        )

        inv = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        pos = np.arange(max_position, dtype=np.float64)[:, None]
        freqs = pos * inv[None, :]
        cos_d = torch.from_numpy(np.cos(freqs).astype(np.float32)).to(torch.bfloat16)
        sin_d = torch.from_numpy(np.sin(freqs).astype(np.float32)).to(torch.bfloat16)
        write_bf16("dflash2.cos_cache", cos_d)
        write_bf16("dflash2.hidden_norm.weight", dflash_loader.read_tensor("hidden_norm.weight"))

        for lid in range(5):
            write_bf16(
                f"dflash2.layers.{lid}.attention_conv.kernel_projection.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.attention_conv.kernel_projection.weight").T,
            )
            write_bf16(
                f"dflash2.layers.{lid}.input_layernorm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.input_layernorm.weight"),
            )
            write_bf16(
                f"dflash2.layers.{lid}.mlp_conv.kernel_projection.MatMul.weight",
                dflash_loader.read_tensor(f"layers.{lid}.mlp_conv.kernel_projection.weight").T,
            )
            write_bf16(
                f"dflash2.layers.{lid}.post_attention_layernorm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.post_attention_layernorm.weight"),
            )
            write_bf16(
                f"dflash2.layers.{lid}.self_attn.k_norm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.self_attn.k_norm.weight"),
            )
            write_bf16(
                f"dflash2.layers.{lid}.self_attn.q_norm.weight",
                dflash_loader.read_tensor(f"layers.{lid}.self_attn.q_norm.weight"),
            )

        write_bf16("dflash2.norm.weight", dflash_loader.read_tensor("norm.weight"))
        write_bf16(
            "dflash2.selector.hidden_projection.MatMul.weight",
            dflash_loader.read_tensor("candidate_selector.hidden_projection.weight").T,
        )
        write_bf16("dflash2.sin_cache", sin_d)
    finally:
        writer.close()

    dflash_proto = build_dflash2_model_proto(
        dflash_locations,
        base_kernels_u16,
        block_size=block_size,
        num_draft_tokens=num_draft_tokens,
        selector_top_k=selector_top_k,
        sliding_window=sliding_window,
    )
    onnx.save_model(dflash_proto, str(out_dir / "dflash2.onnx"), save_as_external_data=False)
    return dflash_locations


@torch.no_grad()
def build_expert_profile_from_hf(
    hf_loader: CheckpointTensorLoader,
    out_path: Path,
    layer_kl_weights: list[float] | None = None,
    num_layers: int = 48,
    num_experts: int = 512,
) -> None:
    """Synthesize STRP v1 `expert-profile.bin` ranking `(layer, expert)` uint16 pairs from router gate norms."""
    scores: list[tuple[float, int, int]] = []
    for lid in range(num_layers):
        g = hf_loader.read_tensor(f"model.language_model.layers.{lid}.mlp.gate.weight").float()
        row_norms = torch.linalg.vector_norm(g, dim=1).tolist()
        lw = float(layer_kl_weights[lid]) if layer_kl_weights is not None and lid < len(layer_kl_weights) else 1.0
        for eid, rn in enumerate(row_norms[:num_experts]):
            scores.append((rn * (1.0 + lw), lid, eid))
    scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    n_ranked = len(scores)
    header = struct.pack("<6I", STRP_MAGIC, 1, num_layers, num_experts, n_ranked, n_ranked)
    entries = np.zeros((n_ranked, 2), dtype=np.uint16)
    for idx, (_sc, lid, eid) in enumerate(scores):
        entries[idx, 0] = lid
        entries[idx, 1] = eid
    out_path.write_bytes(header + entries.tobytes())


def write_weights_pool_txt(out_dir: Path, tensor_locations: dict[str, dict]) -> None:
    """Write `weights_pool.txt` mapping the 5 device-pooled initializers across `target` and `dflash2`."""
    pooled_specs = [
        ("model.rotary_emb.cos_cache", ("target",)),
        ("model.rotary_emb.sin_cache", ("target",)),
        ("capture.indexer_cos_cache.folded", ("target",)),
        ("capture.indexer_sin_cache.folded", ("target",)),
        ("lm_head.weight_t.qweight", ("target", "dflash2")),
    ]
    pool_lines = ["qwen38_weights_pool 1"]
    session_lines: list[str] = []
    for pid, (t_name, sessions) in enumerate(pooled_specs):
        meta = tensor_locations[t_name]
        dims_str = " ".join(str(d) for d in meta["dims"])
        pool_lines.append(
            f"pool {pid} {meta['dtype']} {meta['offset']} {meta['length']} {len(meta['dims'])} {dims_str} {meta['location']}"
        )
        for sess in sessions:
            session_lines.append(f"session {sess} {t_name} {pid}")
    (out_dir / "weights_pool.txt").write_text("\n".join(pool_lines + session_lines) + "\n")


def load_bank_metadata_from_dir(bank_dir: Path) -> tuple[dict[str, dict], dict[str, TensorProto]]:
    """Load `(tensor_locations, embedded_weights)` from `bank_dir/bank_metadata.json` or existing ONNX headers."""
    meta_json = bank_dir / "bank_metadata.json"
    if meta_json.is_file():
        data = json.loads(meta_json.read_text())
        tensor_locations = data["tensor_locations"]
        embedded_weights: dict[str, TensorProto] = {}
        for name, item in data["embedded_weights"].items():
            t = TensorProto()
            t.name = name
            t.data_type = int(item["dtype"])
            t.dims.extend(item["dims"])
            t.raw_data = base64.b85decode(item["raw_b85"])
            embedded_weights[name] = t
        return tensor_locations, embedded_weights

    # Fallback: read initializer metadata from bank_dir/model.onnx and bank_dir/dflash2.onnx
    tensor_locations = {}
    embedded_weights = {}
    for m_name in ("model.onnx", "dflash2.onnx"):
        p = bank_dir / m_name
        if not p.is_file():
            continue
        m = onnx.load(str(p), load_external_data=False)
        for t in m.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if ext and "location" in ext:
                tensor_locations[t.name] = {
                    "location": ext["location"],
                    "offset": int(ext.get("offset", 0)),
                    "length": int(ext.get("length", 0)),
                    "dtype": int(t.data_type),
                    "dims": list(t.dims),
                }
            elif (
                m_name == "model.onnx"
                and ("layers." in t.name or "hyper_connection_mixer" in t.name)
                and t.name != "v_model.layers.0.linear_attn.CastLike_145"
            ):
                t_copy = TensorProto()
                t_copy.CopyFrom(t)
                embedded_weights[t.name] = t_copy
    return tensor_locations, embedded_weights


def save_bank_metadata_to_dir(
    out_dir: Path, tensor_locations: dict[str, dict], embedded_weights: dict[str, TensorProto]
) -> None:
    """Save `bank_metadata.json` so subsequent multi-preset builds can reuse binary `.data` banks without parsing ONNX."""
    non_qmoe_locs = {k: v for k, v in tensor_locations.items() if not str(v.get("location", "")).startswith("qmoe_")}
    emb_serialized = {
        name: {
            "dtype": int(t.data_type),
            "dims": list(t.dims),
            "raw_b85": base64.b85encode(t.raw_data).decode("ascii"),
        }
        for name, t in embedded_weights.items()
    }
    (out_dir / "bank_metadata.json").write_text(
        json.dumps({"tensor_locations": non_qmoe_locs, "embedded_weights": emb_serialized}, indent=2) + "\n"
    )
