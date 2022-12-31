cbuffer buf : register(b0)
{
    float4x4 mvp;
};

void vsMain(float3 position : POSITION,
            float3 normals : NORMAL,
            float2 uv : TEXCOORD,
            out float4 out_position : SV_Position,
            out float2 out_uv : TEXCOORD)
{
    out_uv = uv;
    out_position = mul(float4(position, 1.0), mvp);
}

void psMain(float4 position : SV_Position,
            float2 uv : TEXCOORD,
            out float4 out_color : SV_Target0)
{
    out_color = float4(1.0, 1.0, 1.0, 1.0);
}
