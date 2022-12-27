cbuffer buf : register(b0)
{
    float4x4 mvp;
};

Texture2D tex : register(t0);
SamplerState samp : register(s0);

void vsMain(float3 position : POSITION,
            float2 uv : TEXCOORD0,
            out float4 out_position : SV_Position,
            out float2 out_uv : TEXCOORD0)
{
    out_position = mul(float4(position, 1.0), mvp);
    out_uv = uv;
}

void psMain(float4 position : SV_Position,
            float2 uv : TEXCOORD0,
            out float4 out_color : SV_Target0)
{
    float4 c = tex.Sample(samp, uv);
    out_color = c;
}
