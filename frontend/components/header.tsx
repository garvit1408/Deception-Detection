"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Upload } from "lucide-react"

interface HeaderProps {
  onFileUpload: (file: File) => void
}

export default function Header({ onFileUpload }: HeaderProps) {
  const [language, setLanguage] = useState("Both")
  const [searchQuery, setSearchQuery] = useState("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileUpload(file)
    }
  }

  return (
    <header className="bg-gradient-header shadow-header">
      <div className="container mx-auto px-4 py-4 flex flex-col md:flex-row items-center justify-between">
        <h1 className="text-3xl font-bold text-white mb-4 md:mb-0">Deception Insight AI</h1>

        <div className="flex flex-col md:flex-row gap-3 w-full md:w-auto">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-white">Language</span>
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="w-[120px] bg-white/90 border-0">
                <SelectValue placeholder="Language" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Both">Both</SelectItem>
                <SelectItem value="English">English</SelectItem>
                <SelectItem value="Italian">Italian</SelectItem>
                <SelectItem value="German">German</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Input
            type="text"
            placeholder="Search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full md:w-[300px] bg-white/90 border-0"
          />

          <div className="relative">
            <input
              type="file"
              id="csv-upload"
              accept=".csv"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Button className="w-full md:w-auto bg-white text-insight hover:bg-white/90 hover:text-insight-dark">
              <Upload className="mr-2 h-4 w-4" />
              Upload
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}
