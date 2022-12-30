cbuffer buf : register(b0)
{
    float4x4 mvp;
    float4 color;
};

void vsMain(float3 position : POSITION,
            out float4 out_position : SV_Position)
{
    out_position = mul(float4(position, 1.0), mvp);
}

void psMain(float4 position : SV_Position,
            out float4 out_color : SV_Target0)
{
    out_color = color;
}
