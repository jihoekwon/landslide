# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Chapter 2 of a FastCampus AI Agent course ("Part 1: AI 에이전트와 Claude Code 기초"). This chapter focuses on teaching Claude Code installation and configuration, including MCP server setup and memory configuration.

**Course Information:**
- Instructor: 정구봉 (GB Jeong)
- Course: FastCampus AI Agent & Vibe Coding
- GitBook: https://goobong.gitbook.io/fastcampus
- GitHub: https://github.com/Koomook/fastcampus-ai-agent-vibecoding

## File Structure

```
Chapter2_Claude_Code_설치와_설정/
├── README.md                                      # Chapter overview and learning objectives
├── CLAUDE.md                                       # This file - guidance for Claude Code
├── Clip1_설치와_기본_사용법.md                     # Clip 1: Installation and basic usage
├── Clip2_Local_Project_User_단위_MCP_연결하기.md   # Clip 2: MCP connection at different scopes
└── Clip3_CLAUDE_md_AGENTS_md_세팅하기.md          # Clip 3: Setting up CLAUDE.md and AGENTS.md
```

## Content Structure

This chapter contains educational materials organized into three clips:

1. **Clip1_설치와_기본_사용법.md**: Installation, login, interactive vs one-off modes, execution mode switching (Default, Accept All, Plan Mode, Bypass Permissions)
2. **Clip2_Local_Project_User_단위_MCP_연결하기.md**: MCP server scope levels (Local/Project/User), recommended MCP servers (Playwright, Linear, Codex, Context7), and Smithery.ai marketplace
3. **Clip3_CLAUDE_md_AGENTS_md_세팅하기.md**: Memory management with CLAUDE.md and AGENTS.md files, `/memory`, `/init`, and `#` commands for session memory

## Working with Educational Content

### File Organization

All lecture materials are in Korean and use descriptive filenames following the pattern:
- `ClipN_<topic_description>.md` - Individual lecture clip content
- `README.md` - Chapter overview with learning objectives

### Content Guidelines

When working with these educational materials:

- **Korean Language**: All content is in Korean. Maintain Korean language for any updates or additions to preserve consistency
- **Educational Format**: Content includes step-by-step tutorials, examples, and reference links
- **Instructor Attribution**: All materials include instructor information footer - preserve this when editing
- **GitBook Integration**: Materials are published to GitBook - changes here may need to be synced

### Lecture Material Conventions

- 📋, 🎯, 🗂️, ✅ and other emojis are used for section headers
- Code examples use triple backticks with appropriate language tags
- Step-by-step instructions use `STEP N:` format
- Official documentation links are included in "참고 자료" sections

## Common Tasks

### Viewing Lecture Content
```bash
# Read a specific clip
cat Clip1_설치와_기본_사용법.md

# View chapter overview
cat README.md
```

### Updating Content
When updating educational materials:
1. Maintain Korean language and formatting style
2. Preserve emoji usage patterns for section headers
3. Keep instructor attribution footer intact
4. Update relevant cross-references if content structure changes

### Validating Links
Educational content includes many external reference links (docs.claude.com, smithery.ai, etc.). When updating, verify that:
- Official documentation links are current
- GitBook and GitHub repository links are accurate
- Course URLs remain valid

## Educational Context

### Target Audience
- Korean-speaking developers learning AI agent development
- Students new to Claude Code and MCP servers
- Developers interested in AI-assisted coding workflows

### Learning Path
This chapter (Chapter 2) assumes students have completed Part 1, Chapter 1 and teaches:
1. Installing and configuring Claude Code CLI
2. Understanding and configuring MCP servers at different scope levels
3. Managing AI agent memory through CLAUDE.md and AGENTS.md

### Key Concepts Covered
- **Claude Code Modes**: Interactive vs one-off execution, Default/Accept All/Plan/Bypass Permissions modes
- **MCP Scopes**: Local (personal), Project (team collaboration), User (global utilities)
- **Memory Management**: `/memory`, `/init` commands, `#` symbol for session notes
- **MCP Marketplace**: Using Smithery.ai to discover and install MCP servers
