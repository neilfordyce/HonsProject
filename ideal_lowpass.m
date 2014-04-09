 function Ifilter=ideal_lowpass(I,wlow,whigh)
 % ideally filter image I up to a frequency w

 [N,M]=size(I);
 F=fftshift(fft2(I));
 for i=1:N
     for j=1:N
         r2=(i-round(N/2))^2+(j-round(N/2))^2;
         if (r2<round((N*whigh)^2)) F(i,j)=0; end;
         if (r2>round((N*wlow)^2)) F(i,j)=0; end;
     end;
 end;
 Ifilter=real(ifft2(fftshift(F))); 
 imshow(Ifilter, []);
