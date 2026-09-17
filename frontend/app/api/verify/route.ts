import { NextResponse } from "next/server";
import { verifyClaims } from "@/lib/api";

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "50mb",
    },
  },
};

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const result = await verifyClaims(body);
    return NextResponse.json(result);
  } catch (error: any) {
    console.error("Verification API error:", error);
    return NextResponse.json(
      { error: error.message || "Verification failed" },
      { status: 500 }
    );
  }
}