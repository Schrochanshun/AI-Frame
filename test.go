package main

import (
	"fmt"
	"github.com/lxn/walk"
	. "github.com/lxn/walk/declarative"
	"os/exec"
	"strings"
	"strconv"
)

type Foo struct {
	Bar string
	Baz int
}

var file string

func main() {
	var value int
	var textEdit *walk.TextEdit
	var valueEdit *walk.NumberEdit
	foo := &Foo{"b", 0}

	MainWindow{
		Title:   "AI Frame",
		MinSize: Size{Width: 320, Height: 240},
		Layout:  VBox{},
		DataBinder: DataBinder{
			DataSource: foo,
			AutoSubmit: true,
			OnSubmitted: func() {
				fmt.Println(foo)
			},
		},
		OnDropFiles: func(files []string) {
			textEdit.SetText(strings.Join(files, "\r\n"))
			file = strings.Join(files, "\r\n")
		},
		Children: []Widget{
			TextEdit{
				AssignTo: &textEdit,
				ReadOnly: true,
				Text:     "将mp4文件拖拽至此",
			},
			//
			Composite{
				Layout: Grid{
					Columns: 8,
				},
				Children: []Widget{
					RadioButtonGroup{
						DataMember: "Bar",
						Buttons: []RadioButton{
							RadioButton{
								Name:       "aRB",
								Text:       "X 2",
								Value:      "X 2",
								ColumnSpan: 1,
							},
							RadioButton{
								Name:       "bRB",
								Text:       "X 4",
								Value:      "X 4",
								ColumnSpan: 1,
							},
							RadioButton{
								Name:       "cRB",
								Text:       "X 8",
								Value:      "X 8",
								ColumnSpan: 1,
							},
							RadioButton{
								Name:       "dRD",
								Text:       "自定义帧率",
								ColumnSpan: 1,
							},
						},
					},
					NumberEdit{
						AssignTo:   &valueEdit,
						ColumnSpan: 1,
						Value:      int(value),
						OnValueChanged: func() {
							value = int(valueEdit.Value())
						},
					},
					HSpacer{ColumnSpan: 3},
				},
			},
			PushButton{
				Text: "转化！启动！",
				OnClicked: func() {
					var cmdLine string
					if foo.Bar == "X 2" {
						cmdLine = "python " + "turn.py " + "--exp=1 " + "--video=" + file
					} else if foo.Bar == "X 4" {
						cmdLine = "python " + "turn.py " + "--exp=2 " + "--video=" + file
					} else if foo.Bar == "X 8" {
						cmdLine = "python " + "turn.py " + "--exp=4 " + "--video=" + file
					}	else {
						cmdLine = "python " + "turn.py " + "--video=" + file +" --fps=" +strconv.Itoa(value) 
						fmt.Println(cmdLine)
					}
					fmt.Println(cmdLine)
					cmd := exec.Command("cmd.exe", "/c", "start "+cmdLine)
					fmt.Println(cmdLine)
					cmd.Run()
				},
			},
		},
	}.Run()
}
