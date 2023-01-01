cbuffer buf : register(b0)
{
    float4x4 mvp;
    float4 base_color;
};

Texture2D base_color_tex : register(t0);
SamplerState samp : register(s0);

void vsMain(float3 position : POSITION,
            float3 normals : NORMAL,
            float2 uv : TEXCOORD,
            out float4 out_position : SV_Position,
            out float2 out_uv : TEXCOORD)
{
    out_uv = uv;
    out_position = mul(float4(position, 1.0), mvp);
}

float3 linearToSrgb(float3 color)
{
    float3 S1 = sqrt(color);
    float3 S2 = sqrt(S1);
    float3 S3 = sqrt(S2);
    return 0.585122381 * S1 + 0.783140355 * S2 - 0.368262736 * S3;
}

float4 linearToSrgb(float4 color)
{
    return float4(linearToSrgb(color.rgb), color.a);
}

float3 srgbToLinear(float3 color)
{
    return color * (color * (color * 0.305306011 + 0.682171111) + 0.012522878);
}

float4 srgbToLinear(float4 color)
{
    return float4(srgbToLinear(color.rgb), color.a);
}

void psMain(float4 position : SV_Position,
            float2 uv : TEXCOORD,
            out float4 out_color : SV_Target0)
{
    float4 baseColor = srgbToLinear(base_color);
    float4 baseTexColor = srgbToLinear(base_color_tex.Sample(samp, uv));
    float4 c = baseColor * baseTexColor;
    out_color = linearToSrgb(c);
}
