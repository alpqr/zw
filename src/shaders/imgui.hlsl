cbuffer buf : register(b0)
{
    float4x4 mvp;
};

Texture2D tex : register(t0);
SamplerState samp : register(s0);

void vsMain(float2 position : POSITION,
            float2 uv : TEXCOORD0,
            float4 color : TEXCOORD1,
            out float4 out_position : SV_Position,
            out float2 out_uv : TEXCOORD0,
            out float4 out_color : TEXCOORD1)
{
    out_position = mul(float4(position, 0.0, 1.0), mvp);
    out_uv = uv;
    out_color = color;
}

void psMain(float4 position : SV_Position,
            float2 uv : TEXCOORD0,
            float4 color : TEXCOORD1,
            out float4 out_color : SV_Target0)
{
    out_color = color * tex.Sample(samp, uv);
}
