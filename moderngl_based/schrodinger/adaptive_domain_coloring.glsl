#define TAU (6.283185307179586)
#define pi 3.14159265358979323
// https://shadertoyunofficial.wordpress.com/2019/01/02/programming-tricks-in-shadertoy-glsl/
#define cmod(Z)     length(Z)
#define carg(Z)     atan( (Z).y, (Z).x )
#define csqu(Z)     vec2( (Z).x*(Z).x-(Z).y*(Z).y, -2.0*(Z).x*(Z).y)
#define cmul(A,B) ( mat2( A, -(A).y, (A).x ) * (B) )  // by deMoivre formula
#define cinv(Z)   ( vec2( (Z).x, -(Z).y ) / dot(Z,Z) )
#define cdiv(A,B)   cmul( A, cinv(B) )
#define cpow(Z,v)   pol2cart( vec2( pow(cmod(Z),v) , (v) * carg(Z) ) )
#define cexp(Z)     pol2cart( vec2( exp((Z).x), (Z).y ) )
#define clog(Z)     vec2( log(cmod(Z)), carg(Z) )
#define cis(t)      vec2( cos(t), sin(t))

// Smooth frac
#define sf(x)     smoothstep( .1, .9-step(.4,abs(x-.5)) , x-step(.9,x))
#define sfract(x) sf(fract(x))
// Smooth HSV to RGB conversion (See https://www.shadertoy.com/view/MsS3Wc)
vec3 hsv2rgb( in vec3 c )
{
    vec3 rgb = clamp( abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
	rgb = rgb*rgb*(3.0-2.0*rgb); // cubic smoothing
	return c.z * mix( vec3(1.0), rgb, c.y);
}

float hue2rgb(float f1, float f2, float hue) {
    if (hue < 0.0)
        hue += 1.0;
    else if (hue > 1.0)
        hue -= 1.0;
    float res;
    if ((6.0 * hue) < 1.0)
        res = f1 + (f2 - f1) * 6.0 * hue;
    else if ((2.0 * hue) < 1.0)
        res = f2;
    else if ((3.0 * hue) < 2.0)
        res = f1 + (f2 - f1) * ((2.0 / 3.0) - hue) * 6.0;
    else
        res = f1;
    return res;
}

vec3 hsl2rgb(vec3 hsl) {
    vec3 rgb;

    if (hsl.y == 0.0) {
        rgb = vec3(hsl.z); // Luminance
    } else {
        float f2;

        if (hsl.z < 0.5)
            f2 = hsl.z * (1.0 + hsl.y);
        else
            f2 = hsl.z + hsl.y - hsl.y * hsl.z;

        float f1 = 2.0 * hsl.z - f2;

        rgb.r = hue2rgb(f1, f2, hsl.x + (1.0/3.0));
        rgb.g = hue2rgb(f1, f2, hsl.x);
        rgb.b = hue2rgb(f1, f2, hsl.x - (1.0/3.0));
    }
    return rgb;
}

vec4 imagineColor(vec2 z){
    // Let's grab the angle and magnitude for the number
    float arg = atan(z.y, z.x);
    float mag = length(z);

    // We can base the Hue on the angle of the number,
    // and use a nice light Saturation. Full Value, so
    // we can be sure to see it nice and bright.
    vec3 hsv = vec3(arg/TAU,1.0,1.0);

    // But actually... it would be nice to know
    // where some special values are. You know, the
    // integer Real, integer Imaginary; nice Magnitudes...
    // We'll carve out Blocks and Rings for those numbers
    // by multiplying our hsv by 0s.


    // First, figure out where the magnitude is a power of 2.
    // fract(x) ramps up from 0 to 1 repeatedly -- it's
    // the same as mod(x,1), or the "fractional part" of
    // the number.
    //float rings = fract(log2(mag));
    // ... but that's a bit too harsh of a ramp, so
    // we smooth it out. And then smooth it out again.
    //	  rings = smoothstep(0.,0.4*2.,1.-abs(2.*rings-1.));
    //	  rings = smoothstep(0.,0.08,rings);
    // c.f. https://www.desmos.com/calculator/xs5xqge9hj

    // I want my rings to be white, so I'll multiply
    // into the 'saturation' component of hsv.
    //hsv.y *= rings;

    // To draw my blocks, I'll peak at the fractional
    // part of each component. If I scale up by 2, I can
    // draw on the half-integer marks.
    //vec2 blocks = fract(z*2.0);
    // Again, too wide a transition, so we'll compress it down.
    //	 blocks = smoothstep(0.,0.05*2.,1.-abs(2.*blocks-1.));

    // The blocks will be black, so we'll multiply against
    // the 'value' component of hsv.
    //hsv.z *= blocks.x*blocks.y;

//    hsv.y *=1 - pow(2.0, -mag/ 1);
    float h = (arg + pi)  / (2 * pi) + 0.5;

    float l = 1.0 - 1.0/(1.0 + sqrt(sqrt((z.x*z.x+z.y*z.y))));
    float s = 0.8;

    vec3 hsl = vec3(h,s,l);

    return vec4(hsl2rgb(hsl), 1.0);
//    return vec4(hsv2rgb(hsv) * (mag*mag),1.0);//min(1,mag), 1.0);
}