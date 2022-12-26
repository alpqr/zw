cbuffer buf : register(b0)
{
    float4x4 mvp;
};

void vsMain(float3 position : POSITION,
            float3 color : TEXCOORD0,
            out float4 out_position : SV_Position,
            out float3 out_color : TEXCOORD0)
{
    out_position = mul(mvp, float4(position, 1.0));
    out_color = color;
}

void psMain(float4 position : SV_Position,
            float3 color : TEXCOORD0,
            out float4 out_color : SV_Target0)
{
    out_color = float4(color, 1.0);
}
