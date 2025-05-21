import { type NextRequest, NextResponse } from "next/server"
import { analyzeConversations } from "@/lib/analysis"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Check if it's a CSV file
    if (!file.name.endsWith(".csv")) {
      return NextResponse.json({ error: "Only CSV files are supported" }, { status: 400 })
    }

    // Read the file content
    const text = await file.text()
    const lines = text.split("\n")
    const headers = lines[0].split(",")

    // Parse CSV
    const parsedData = lines.slice(1).map((line) => {
      const values = line.split(",")
      const row: Record<string, string> = {}
      headers.forEach((header, index) => {
        row[header.trim()] = values[index]?.trim() || ""
      })
      return row
    })

    // Analyze the data
    const conversations = analyzeConversations(parsedData)

    return NextResponse.json({
      success: true,
      conversations,
      message: `Successfully processed ${conversations.length} conversations`,
    })
  } catch (error) {
    console.error("Error processing file:", error)
    return NextResponse.json({ error: "Failed to process file" }, { status: 500 })
  }
}
